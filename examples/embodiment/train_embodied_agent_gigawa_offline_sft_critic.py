import json
import os
import time

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


def _pick_group_metrics(metrics):
    if isinstance(metrics, list):
        for item in metrics:
            if isinstance(item, dict):
                return item
        return {}
    return metrics if isinstance(metrics, dict) else {}


def _fmt_metric(value):
    if value is None:
        return "NA"
    try:
        return f"{float(value):.8f}"
    except Exception:
        return str(value)


def _save_actor_checkpoint(actor_group, cfg, global_step: int):
    base_output_dir = os.path.join(
        cfg.runner.logger.log_path,
        cfg.runner.logger.experiment_name,
        f"checkpoints/global_step_{global_step}",
    )
    actor_save_path = os.path.join(base_output_dir, "actor")
    os.makedirs(actor_save_path, exist_ok=True)
    actor_group.save_checkpoint(actor_save_path, global_step).wait()
    return actor_save_path


def _load_resume_checkpoint(actor_group, cfg) -> int:
    resume_dir = cfg.runner.get("resume_dir", None)
    global_step = 0
    if not resume_dir:
        return global_step

    actor_resume_path = (
        os.path.join(resume_dir, "actor")
        if os.path.isdir(os.path.join(resume_dir, "actor"))
        else resume_dir
    )
    actor_group.load_checkpoint(actor_resume_path).wait()
    if "global_step_" in resume_dir:
        try:
            global_step = int(resume_dir.split("global_step_")[-1].split("/")[0])
        except Exception:
            global_step = 0
    return global_step


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="offline_sft_critic_pretrain",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.get("per_worker_log_path", None)
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_placement = component_placement.get_strategy("actor")

    from rlinf.workers.actor.fsdp_gigawa_policy_worker import EmbodiedGigaWAFSDPPolicy

    actor_group = EmbodiedGigaWAFSDPPolicy.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    actor_group.init_offline_sft_critic_worker().wait()
    global_step = _load_resume_checkpoint(actor_group, cfg)

    tb_dir = os.path.join(
        cfg.runner.logger.log_path,
        cfg.runner.logger.experiment_name,
        "tensorboard",
    )
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    max_epochs = int(cfg.runner.max_epochs)
    save_interval = int(cfg.runner.get("save_interval", 100))
    critic_cfg = cfg.algorithm.get("offline_critic_pretrain", {})
    val_interval = max(1, int(critic_cfg.get("val_interval", 5)))
    class_eval_interval = max(1, int(critic_cfg.get("class_eval_interval", val_interval)))
    tb_flush_interval = max(1, int(critic_cfg.get("tb_flush_interval", val_interval)))
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        do_validation = (epoch == 1) or (epoch % val_interval == 0) or (epoch == max_epochs)
        do_class_eval = (epoch == 1) or (epoch % class_eval_interval == 0) or (epoch == max_epochs)

        bc_metrics = _pick_group_metrics(actor_group.run_offline_bc_epoch().wait())
        critic_metrics = _pick_group_metrics(
            actor_group.run_offline_critic_epoch(
                do_validation=do_validation,
                do_class_eval=do_class_eval,
            ).wait()
        )
        metrics = {}
        metrics.update(bc_metrics)
        metrics.update(critic_metrics)

        global_step += 1
        elapsed = time.time() - start_time

        print(
            f"[offline_sft_critic] epoch={epoch:04d} | step={global_step:06d} | elapsed={elapsed:.1f}s | "
            f"sft_train_bc={_fmt_metric(metrics.get('offline_bc/train_bc_loss'))} | "
            f"sft_val_bc={_fmt_metric(metrics.get('offline_bc/val_bc_loss'))} | "
            f"critic_train_loss={_fmt_metric(metrics.get('offline_critic/train_critic_loss'))} | "
            f"critic_val_loss={_fmt_metric(metrics.get('offline_critic/val_critic_loss'))} | "
            f"train_q={_fmt_metric(metrics.get('offline_critic/train_q_logged_mean'))} | "
            f"val_q={_fmt_metric(metrics.get('offline_critic/val_q_logged_mean'))} | "
            f"train_sf_gap={_fmt_metric(metrics.get('offline_critic/train_success_failure_q_gap'))} | "
            f"val_sf_gap={_fmt_metric(metrics.get('offline_critic/val_success_failure_q_gap'))} | "
            f"actor_grad={_fmt_metric(metrics.get('actor/grad_norm'))} | "
            f"critic_grad={_fmt_metric(metrics.get('critic/grad_norm'))}"
        )
        if do_class_eval:
            print(
                "[offline_sft_critic][success_failure] "
                f"train_success_q={_fmt_metric(metrics.get('offline_critic/train_success/q_logged_mean'))} | "
                f"train_failure_q={_fmt_metric(metrics.get('offline_critic/train_failure/q_logged_mean'))} | "
                f"val_success_q={_fmt_metric(metrics.get('offline_critic/val_success/q_logged_mean'))} | "
                f"val_failure_q={_fmt_metric(metrics.get('offline_critic/val_failure/q_logged_mean'))}"
            )

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, global_step)
        writer.add_scalar("offline_sft_critic/epoch", epoch, global_step)
        if (
            (epoch % tb_flush_interval == 0)
            or do_validation
            or do_class_eval
            or (epoch % save_interval == 0)
            or (epoch == max_epochs)
        ):
            writer.flush()

        if epoch % save_interval == 0 or epoch == max_epochs:
            save_path = _save_actor_checkpoint(actor_group, cfg, global_step)
            print(f"[offline_sft_critic] saved checkpoint: {save_path}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
