# Copyright 2026 The GIGA Authors.
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

"""Piper 双臂插桩类任务环境（关节增量 + 可选主臂人在回路）。

与 ``Franka PegInsertionEnv`` 在 RLinf 中的用法对齐：通过 ``override_cfg`` 注入
``PiperRobotConfig`` 字段；策略侧使用 ``joint_action_mode=delta`` 时输出
``[-1,1]^14``，环境内乘 ``delta_action_scale`` 再累加到当前 qpos。

奖励与成功条件可按真机标定在此子类中扩展。
"""

from dataclasses import dataclass

from ..piper_env import PiperEnv, PiperRobotConfig


@dataclass
class PiperPegInsertionConfig(PiperRobotConfig):
    """插桩任务默认：增量关节、主臂介入开启（与 ``MasterArmController`` 配合）。"""

    task_name: str = "piper_peg_insertion"
    max_num_steps: int = 1000
    joint_action_mode: str = "delta"
    delta_action_scale: float = 0.05
    enable_human_intervention: bool = True


class PiperPegInsertionEnv(PiperEnv):
    """Piper 双臂 peg-insertion 模板环境。"""

    def __init__(
        self,
        override_cfg: dict | None = None,
        worker_info=None,
        hardware_info=None,
        env_idx: int = 0,
    ) -> None:
        if override_cfg is None:
            override_cfg = {}
        config = PiperPegInsertionConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self) -> str:
        return "peg and insertion"
