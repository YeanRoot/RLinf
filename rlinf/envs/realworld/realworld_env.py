# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import pathlib
import time
from functools import partial
from typing import OrderedDict

import gymnasium as gym
import numpy as np
import psutil
import torch
from filelock import FileLock
from omegaconf import OmegaConf

from rlinf.envs.realworld.common.wrappers import (
    GripperCloseEnv,
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
    Quat2EulerWrapper,
    RelativeFrame,
    SpacemouseIntervention,
)
from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv
from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo


class RealWorldEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        assert num_envs == 1, (
            f"Currently, only 1 realworld env can be started per worker, but {num_envs=} is received."
        )

        self.cfg = cfg
        self.override_cfg = OmegaConf.to_container(
            cfg.get("override_cfg", OmegaConf.create({})), resolve=True
        )

        self.video_cfg = cfg.video_cfg

        self.seed = cfg.seed + seed_offset
        self.num_envs = num_envs
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.main_image_key = cfg.main_image_key
        wrist_image_keys = cfg.get("wrist_image_keys", None)
        self.wrist_image_keys = list(wrist_image_keys) if wrist_image_keys else []
        self.manual_episode_control_only = bool(
            self.override_cfg.get("manual_episode_control_only", False)
        )
        self.break_chunk_on_intervention = bool(
            cfg.get("break_chunk_on_intervention", False)
        )
        # New real-robot corrective relabeling mode:
        # once intervention starts inside a chunk, latch it until the current
        # chunk ends; then automatically release teleop and let the next chunk
        # be planned from the latest observation.
        self.latch_intervention_until_chunk_end = bool(
            cfg.get("latch_intervention_until_chunk_end", False)
        )
        # Kept for backward compatibility with older configs.  When
        # latch_intervention_until_chunk_end=True this is intentionally ignored.
        self.collect_intervention_until_release = bool(
            cfg.get("collect_intervention_until_release", False)
        )
        self.pad_interrupted_chunks = bool(cfg.get("pad_interrupted_chunks", True))
        self.intervention_max_steps = int(cfg.get("intervention_max_steps", 96))
        self.debug_intervention_chunks = bool(cfg.get("debug_intervention_chunks", True))
        self.force_disable_teleop_on_chunk_end = bool(
            cfg.get("force_disable_teleop_on_chunk_end", True)
        )
        self.force_disable_teleop_on_terminal = bool(
            cfg.get("force_disable_teleop_on_terminal", True)
        )
        self.force_disable_teleop_on_timeout = bool(
            cfg.get("force_disable_teleop_on_timeout", True)
        )
        self._last_raw_states = None

        self._init_env()

        self._is_start = True
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._init_reset_state_ids()

    def _create_env(self, env_idx: int):
        worker_info: WorkerInfo = self.worker_info
        hardware_info = None
        if worker_info is not None and env_idx < len(worker_info.hardware_infos):
            hardware_info = worker_info.hardware_infos[env_idx]
        override_cfg = copy.deepcopy(self.override_cfg)
        env = gym.make(
            id=self.cfg.init_params.id,
            override_cfg=override_cfg,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
            env_cfg=self.cfg,
        )
        stack = self.cfg.get("realworld_stack", "franka")

        if stack == "piper_joint":
            # Piper 双臂关节空间：无 tcp_pose；不套 GripperCloseEnv（仅适用于 Franka 7D EE）
            if not env.config.is_dummy and self.cfg.get("use_spacemouse", False):
                env = SpacemouseIntervention(env)
            if not env.config.is_dummy and self.cfg.get("keyboard_reward_wrapper", None):
                if self.cfg.keyboard_reward_wrapper == "multi_stage":
                    env = KeyboardRewardDoneMultiStageWrapper(env)
                elif self.cfg.keyboard_reward_wrapper == "single_stage":
                    env = KeyboardRewardDoneWrapper(env)
            return env

        # ---- Franka 默认栈 ----
        if self.cfg.get("no_gripper", True):
            env = GripperCloseEnv(env)
        if not env.config.is_dummy and self.cfg.get("use_spacemouse", True):
            env = SpacemouseIntervention(env)
        if not env.config.is_dummy and self.cfg.get("keyboard_reward_wrapper", None):
            if self.cfg.keyboard_reward_wrapper == "multi_stage":
                env = KeyboardRewardDoneMultiStageWrapper(env)
            elif self.cfg.keyboard_reward_wrapper == "single_stage":
                env = KeyboardRewardDoneWrapper(env)

        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        return env

    @staticmethod
    def realworld_setup():
        """Setup RealWorld environment upon env class import.

        This is for any node-level setup required by RealWorld environments. For example, ROS
        requires a single roscore instance per node, so we ensure that any existing roscore
        processes are terminated before starting a new one.

        This function is called once when the RealWorldEnv class is first imported.
        
        Set RLINF_SKIP_ROS_CLEANUP=1 to skip ROS cleanup if you want to keep existing ROS nodes.
        """
        if os.environ.get("RLINF_SKIP_ROS_CLEANUP", "0") == "1":
            return
        
        # Concurrency control is needed for multiple processes on the same node
        node_lock_file = "/tmp/.realworld.lock"
        # Check if the path is valid
        if not os.path.exists(os.path.dirname(node_lock_file)):
            node_lock_file = os.path.join(pathlib.Path.home(), ".realworld.lock")
        node_lock = FileLock(node_lock_file)

        with node_lock:
            ros_proc_names = ["roscore", "rosmaster", "rosout"]
            for proc in psutil.process_iter():
                if proc.name() in ros_proc_names:
                    proc.kill()
                    time.sleep(0.5)

    def _init_env(self):
        env_fns = [
            partial(self._create_env, env_idx=env_idx)
            for env_idx in range(self.num_envs)
        ]
        self.env = NoAutoResetSyncVectorEnv(env_fns)
        self.task_descriptions = list(
            self.env.call("get_wrapper_attr", "task_description")
        )

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def total_num_group_envs(self):
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _init_metrics(self):
        self.prev_step_reward = np.zeros(self.num_envs)

        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[mask] = 0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(
        self,
        step_reward,
        terminations,
        success_current_step,
        intervene_current_step,
        infos,
    ):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | success_current_step
        self.intervened_once = self.intervened_once | intervene_current_step
        self.intervened_steps += intervene_current_step.astype(int)

        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["intervened_once"] = self.intervened_once
        episode_info["intervened_steps"] = self.intervened_steps
        episode_info["success_no_intervened"] = self.success_once.copy() & (
            ~self.intervened_once
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _raw_state_from_raw_obs(self, raw_obs):
        raw_states = OrderedDict(sorted(raw_obs["state"].items()))
        return np.concatenate([value for value in raw_states.values()], axis=-1)

    def _hold_action_from_obs(self, obs):
        # obs is wrapped tensor obs; use current raw qpos to safely hold position.
        states = obs.get("states", None) if isinstance(obs, dict) else None
        if isinstance(states, torch.Tensor):
            return states.detach().cpu().numpy().copy()
        if states is not None:
            return np.asarray(states).copy()
        if self._last_raw_states is not None:
            return self._last_raw_states.copy()
        return np.zeros((self.num_envs, 14), dtype=np.float64)

    def _force_disable_teleop_for_all_envs(self, reason: str) -> None:
        """Best-effort teleop release for all vectorized real-world envs."""
        envs = getattr(self.env, "envs", [])
        for env in envs:
            called = False
            for target in (env, getattr(env, "unwrapped", None)):
                if target is None:
                    continue
                fn = getattr(target, "force_disable_teleop", None)
                if callable(fn):
                    fn(reason=reason)
                    called = True
                    break
            if not called and self.debug_intervention_chunks:
                print(
                    f"[RealWorldEnv] force_disable_teleop unavailable for env={type(env).__name__}, reason={reason}",
                    flush=True,
                )

    def _pad_tensor_list(self, values, target_len, *, fill_like=None, fill_value=0):
        if not values:
            raise RuntimeError("Cannot pad an empty tensor list.")
        out = list(values)
        ref = fill_like if fill_like is not None else values[-1]
        while len(out) < target_len:
            out.append(torch.full_like(ref, fill_value))
        return out

    def reset(self, *, reset_state_ids=None, seed=None, options=None, env_idx=None):
        # TODO: handle partial reset
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        self._last_raw_states = self._raw_state_from_raw_obs(raw_obs).copy()

        extracted_obs = self._wrap_obs(raw_obs)
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, infos

    def _wrap_obs(self, raw_obs):
        """
        raw_obs: Dict of list
        """
        obs = {}

        # Process states
        full_states = []
        raw_states = OrderedDict(sorted(raw_obs["state"].items()))
        for value in raw_states.values():
            full_states.append(value)
        full_states = np.concatenate(full_states, axis=-1)
        obs["states"] = full_states

        frames = raw_obs["frames"]
        if self.main_image_key not in frames:
            raise KeyError(
                f"main_image_key {self.main_image_key!r} not in {list(frames)}"
            )
        obs["main_images"] = frames[self.main_image_key]

        if self.wrist_image_keys:
            missing_wrist_keys = [
                key for key in self.wrist_image_keys if key not in frames
            ]
            if missing_wrist_keys:
                raise KeyError(
                    f"wrist_image_keys {missing_wrist_keys!r} not in {list(frames)}"
                )
            # Keep the order from config so GigaWorldPolicy sees
            # [left_wrist, right_wrist], matching WA training preprocessing.
            obs["wrist_images"] = np.stack(
                [frames[key] for key in self.wrist_image_keys], axis=1
            )

        raw_images = OrderedDict(sorted(frames.items()))
        raw_images.pop(self.main_image_key)

        if raw_images:
            obs["extra_view_images"] = np.stack(list(raw_images.values()), axis=1)

        obs = to_tensor(obs)
        obs["task_descriptions"] = self.task_descriptions
        return obs

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        if actions is None:
            actions = self._last_raw_states.copy() if self._last_raw_states is not None else np.zeros((self.num_envs, 14), dtype=np.float64)

        policy_actions_abs = np.asarray(actions, dtype=np.float64).copy()
        raw_states_before = (
            self._last_raw_states.copy()
            if self._last_raw_states is not None
            else np.zeros_like(policy_actions_abs, dtype=np.float64)
        )

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        raw_states_after = self._raw_state_from_raw_obs(raw_obs).copy()
        self._last_raw_states = raw_states_after.copy()
        timeout_truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        if not self.manual_episode_control_only:
            truncations = timeout_truncations

        # Terminal events must release teleop immediately.  Otherwise reset/policy
        # commands can keep fighting the master-arm teleop node.  Failure/timeout
        # still has reward 0; this only changes control ownership.
        if bool(np.asarray(terminations).any()) and self.force_disable_teleop_on_terminal:
            self._force_disable_teleop_for_all_envs(reason="terminal")
        if bool(np.asarray(truncations).any()) and self.force_disable_teleop_on_timeout:
            self._force_disable_teleop_for_all_envs(reason="timeout")

        obs = self._wrap_obs(raw_obs)
        step_reward = self._calc_step_reward(_reward)
        success_current_step = np.isclose(step_reward, 1.0)
        teleop_active = np.zeros(self.num_envs, dtype=bool)
        if "teleop_active" in infos:
            for env_id in range(self.num_envs):
                teleop_active[env_id] = bool(infos["teleop_active"][env_id])

        intervene_flag = np.zeros(self.num_envs, dtype=bool)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                if infos["intervene_action"][env_id] is not None:
                    intervene_flag[env_id] = True
        intervene_flag = intervene_flag | teleop_active

        infos = self._record_metrics(
            step_reward,
            terminations,
            success_current_step,
            intervene_flag,
            infos,
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        intervene_action = np.zeros_like(policy_actions_abs)
        executed_action_abs = policy_actions_abs.copy()
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                env_intervene_action = infos["intervene_action"][env_id]
                if env_intervene_action is not None:
                    intervene_action[env_id] = env_intervene_action.copy()
                    executed_action_abs[env_id] = env_intervene_action.copy()
        if "executed_action_abs" in infos:
            for env_id in range(self.num_envs):
                env_executed_action = infos["executed_action_abs"][env_id]
                if env_executed_action is not None:
                    executed_action_abs[env_id] = env_executed_action.copy()

        infos["intervene_action"] = to_tensor(intervene_action)
        infos["intervene_flag"] = to_tensor(intervene_flag)
        infos["teleop_active"] = to_tensor(teleop_active)
        infos["policy_action_abs"] = to_tensor(policy_actions_abs)
        infos["executed_action_abs"] = to_tensor(executed_action_abs)
        infos["raw_state_before_action"] = to_tensor(raw_states_before)
        infos["raw_state_after_action"] = to_tensor(raw_states_after)

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        """Execute a model action chunk with chunk-latched intervention.

        Real-robot intervention semantics:
        - If no intervention occurs, execute the full model chunk.
        - Once intervention is detected at primitive step k, the remainder of the
          current chunk is treated as human correction.  We stop consuming stale
          model actions and feed a hold action to the env; while teleop is active,
          PiperEnv ignores that command and records the actual puppet action.
        - At the end of the chunk, teleop is force-disabled so the next chunk is
          automatically planned/executed by the model from the latest observation.
        - If success/failure/timeout happens inside the chunk, remaining entries
          are padded with action_valid_mask=False.
        """
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = int(chunk_actions.shape[1])
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        raw_chunk_intervene_actions = []
        raw_chunk_intervene_flag = []
        raw_chunk_valid_mask = []
        raw_chunk_policy_actions_abs = []
        raw_chunk_executed_actions_abs = []
        raw_chunk_raw_states_before = []

        interrupted = False
        intervention_latched = False
        intervention_start_step = None
        latest_obs = None
        latest_infos = None

        def _force_intervention_suffix_info(infos):
            """Make latched suffix explicit even if teleop briefly flickers off."""
            if "intervene_flag" in infos:
                infos["intervene_flag"] = torch.ones_like(
                    infos["intervene_flag"], dtype=torch.bool
                )
            if "intervene_action" in infos and "executed_action_abs" in infos:
                # Keep training/debug contracts aligned: intervention action is
                # exactly the executed env-space action for latched suffix steps.
                infos["intervene_action"] = infos["executed_action_abs"].clone()

        def _append_record(extracted_obs, step_reward, terminations, truncations, infos, *, valid=True):
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)
            raw_chunk_intervene_actions.append(infos["intervene_action"])
            raw_chunk_intervene_flag.append(infos["intervene_flag"])
            raw_chunk_valid_mask.append(
                torch.ones_like(infos["intervene_flag"], dtype=torch.bool)
                if valid else torch.zeros_like(infos["intervene_flag"], dtype=torch.bool)
            )
            raw_chunk_policy_actions_abs.append(infos["policy_action_abs"])
            raw_chunk_executed_actions_abs.append(infos["executed_action_abs"])
            raw_chunk_raw_states_before.append(infos["raw_state_before_action"])

        for i in range(chunk_size):
            # Once intervention is latched, never execute the stale remainder of
            # the model chunk.  PiperEnv ignores this hold command while teleop is
            # active and records the true puppet action instead.
            if intervention_latched:
                actions = self._hold_action_from_obs(latest_obs)
            else:
                actions = chunk_actions[:, i]

            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            latest_obs, latest_infos = extracted_obs, infos

            intervention_now = bool(infos["intervene_flag"].any().item())
            if (
                intervention_now
                and self.break_chunk_on_intervention
                and self.latch_intervention_until_chunk_end
                and not intervention_latched
            ):
                intervention_latched = True
                interrupted = True
                intervention_start_step = i

            if intervention_latched:
                _force_intervention_suffix_info(infos)

            _append_record(extracted_obs, step_reward, terminations, truncations, infos, valid=True)

            done_now = bool(torch.logical_or(terminations, truncations).any().item())
            if done_now:
                break

            if (
                intervention_now
                and self.break_chunk_on_intervention
                and not self.latch_intervention_until_chunk_end
            ):
                # Legacy behavior: terminate the current chunk when intervention starts.
                interrupted = True
                break

        if not obs_list:
            raise RuntimeError("RealWorldEnv.chunk_step produced no environment steps.")

        valid_steps = len(chunk_rewards)

        # If the chunk completed normally after a latched intervention, release
        # teleop so the next chunk is generated/executed by the model.
        last_done = bool(
            torch.logical_or(raw_chunk_terminations[-1], raw_chunk_truncations[-1]).any().item()
        )
        if (
            intervention_latched
            and not last_done
            and self.force_disable_teleop_on_chunk_end
        ):
            self._force_disable_teleop_for_all_envs(reason="chunk_end")

        if self.pad_interrupted_chunks and valid_steps < chunk_size:
            pad_count = chunk_size - valid_steps
            last_obs = latest_obs if latest_obs is not None else obs_list[-1]
            last_infos = latest_infos if latest_infos is not None else infos_list[-1]
            zero_reward = torch.zeros_like(chunk_rewards[-1])
            false_term = torch.zeros_like(raw_chunk_terminations[-1], dtype=torch.bool)
            false_trunc = torch.zeros_like(raw_chunk_truncations[-1], dtype=torch.bool)
            for _ in range(pad_count):
                obs_list.append(last_obs)
                infos_list.append(last_infos)
                chunk_rewards.append(zero_reward.clone())
                raw_chunk_terminations.append(false_term.clone())
                raw_chunk_truncations.append(false_trunc.clone())
                raw_chunk_intervene_actions.append(torch.zeros_like(raw_chunk_intervene_actions[-1]))
                raw_chunk_intervene_flag.append(torch.zeros_like(raw_chunk_intervene_flag[-1], dtype=torch.bool))
                raw_chunk_valid_mask.append(torch.zeros_like(raw_chunk_valid_mask[-1], dtype=torch.bool))
                raw_chunk_policy_actions_abs.append(raw_chunk_policy_actions_abs[-1].clone())
                raw_chunk_executed_actions_abs.append(raw_chunk_executed_actions_abs[-1].clone())
                raw_chunk_raw_states_before.append(raw_chunk_raw_states_before[-1].clone())

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_terminations.any() and self.force_disable_teleop_on_terminal:
            self._force_disable_teleop_for_all_envs(reason="terminal_chunk")
        if past_truncations.any() and self.force_disable_teleop_on_timeout:
            self._force_disable_teleop_for_all_envs(reason="timeout_chunk")

        infos_last = infos_list[-1] if infos_list else {}
        infos_last["intervene_action"] = torch.stack(
            raw_chunk_intervene_actions, dim=1
        ).reshape(self.num_envs, -1)
        infos_last["intervene_flag"] = torch.stack(raw_chunk_intervene_flag, dim=1)
        infos_last["action_valid_mask"] = torch.stack(raw_chunk_valid_mask, dim=1)
        infos_last["policy_action_abs"] = torch.stack(
            raw_chunk_policy_actions_abs, dim=1
        ).reshape(self.num_envs, -1)
        infos_last["executed_action_abs"] = torch.stack(
            raw_chunk_executed_actions_abs, dim=1
        ).reshape(self.num_envs, -1)
        infos_last["raw_state_before_action"] = torch.stack(
            raw_chunk_raw_states_before, dim=1
        )
        infos_last["chunk_interrupted"] = to_tensor(
            np.asarray([interrupted] * self.num_envs, dtype=bool)
        )
        infos_last["intervention_latched"] = to_tensor(
            np.asarray([intervention_latched] * self.num_envs, dtype=bool)
        )
        infos_last["intervention_start_step"] = to_tensor(
            np.asarray([
                -1 if intervention_start_step is None else int(intervention_start_step)
            ] * self.num_envs, dtype=np.int64)
        )
        infos_last["executed_steps"] = to_tensor(
            np.asarray([valid_steps] * self.num_envs, dtype=np.int64)
        )
        infos_list[-1] = infos_last

        if self.debug_intervention_chunks and interrupted:
            print(
                "[RealWorldEnv INTERVENTION] "
                f"mode={'latch_to_chunk_end' if self.latch_intervention_until_chunk_end else 'legacy'}, "
                f"start_step={intervention_start_step}, "
                f"valid_steps={valid_steps}/{chunk_size}, "
                f"intervene_flags={infos_last['intervene_flag'].cpu().numpy().astype(int).tolist()}, "
                f"valid_mask={infos_last['action_valid_mask'].cpu().numpy().astype(int).tolist()}",
                flush=True,
            )

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=(
                self.reset_state_ids[env_idx]
                if self.use_fixed_reset_state_ids
                else None
            ),
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, reward: np.ndarray):
        return reward.astype(np.float32)

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        )
