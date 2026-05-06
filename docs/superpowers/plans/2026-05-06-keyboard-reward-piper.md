# Keyboard Reward for Piper Online RL Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sync `reward_done_wrapper.py` with RLinf_jie (fix noisy print statements) and enable keyboard reward for Piper online RL training via config.

**Architecture:** Two isolated changes — (1) a one-line conditional print fix in two methods of the existing wrapper, and (2) one new YAML key in the Piper env config. No structural changes. `realworld_env.py` already routes `keyboard_reward_wrapper` for the `piper_joint` stack.

**Tech Stack:** Python 3, pytest, gymnasium, PyYAML (config only)

---

## File Map

| Action | File | What changes |
|--------|------|--------------|
| Modify | `rlinf/envs/realworld/common/wrappers/reward_done_wrapper.py` | Wrap `print` with `if key is not None` in two methods |
| Modify | `examples/embodiment/config/env/realworld_piper_collect.yaml` | Add `keyboard_reward_wrapper: single_stage` |
| Create | `tests/unit_tests/envs/realworld/test_keyboard_reward_done_wrapper.py` | Unit tests for wrapper behavior |

---

### Task 1: Write failing tests for print behavior and reward logic

**Files:**
- Create: `tests/unit_tests/envs/realworld/test_keyboard_reward_done_wrapper.py`

- [ ] **Step 1: Create the test directory and file**

```bash
mkdir -p tests/unit_tests/envs/realworld
touch tests/unit_tests/envs/realworld/__init__.py
```

- [ ] **Step 2: Write the tests**

`KeyboardRewardDoneWrapper` and `KeyboardRewardDoneMultiStageWrapper` both depend on `KeyboardListener`, which requires Linux `evdev`. We mock `self.listener` directly on the wrapper instance to avoid the hardware dependency.

Create `tests/unit_tests/envs/realworld/test_keyboard_reward_done_wrapper.py`:

```python
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

"""Unit tests for KeyboardRewardDoneWrapper and KeyboardRewardDoneMultiStageWrapper.

KeyboardListener requires Linux evdev — we mock self.listener on each wrapper
instance so these tests run on any platform.
"""

from unittest.mock import MagicMock, patch

import pytest


def _make_single_stage_wrapper():
    """Return a KeyboardRewardDoneWrapper with a mocked listener and env."""
    with patch(
        "rlinf.envs.realworld.common.wrappers.reward_done_wrapper.KeyboardListener"
    ):
        from rlinf.envs.realworld.common.wrappers.reward_done_wrapper import (
            KeyboardRewardDoneWrapper,
        )

        env = MagicMock()
        wrapper = KeyboardRewardDoneWrapper(env)
    wrapper.listener = MagicMock()
    return wrapper


def _make_multi_stage_wrapper():
    """Return a KeyboardRewardDoneMultiStageWrapper with a mocked listener and env."""
    with patch(
        "rlinf.envs.realworld.common.wrappers.reward_done_wrapper.KeyboardListener"
    ):
        from rlinf.envs.realworld.common.wrappers.reward_done_wrapper import (
            KeyboardRewardDoneMultiStageWrapper,
        )

        env = MagicMock()
        wrapper = KeyboardRewardDoneMultiStageWrapper(env)
    wrapper.listener = MagicMock()
    wrapper.reward_stage = 0
    return wrapper


# ---------------------------------------------------------------------------
# KeyboardRewardDoneWrapper — reward/done logic
# ---------------------------------------------------------------------------


class TestKeyboardRewardDoneWrapper:
    def test_no_key_returns_zero_reward_not_done(self):
        wrapper = _make_single_stage_wrapper()
        wrapper.listener.get_key.return_value = None
        intervened, done, reward = wrapper._check_keypress()
        assert reward == 0
        assert done is False
        assert intervened is False

    def test_key_a_returns_negative_reward_and_done(self):
        wrapper = _make_single_stage_wrapper()
        wrapper.listener.get_key.return_value = "a"
        intervened, done, reward = wrapper._check_keypress()
        assert reward == -1
        assert done is True
        assert intervened is True

    def test_key_b_returns_zero_reward_not_done(self):
        wrapper = _make_single_stage_wrapper()
        wrapper.listener.get_key.return_value = "b"
        intervened, done, reward = wrapper._check_keypress()
        assert reward == 0
        assert done is False
        assert intervened is True

    def test_key_c_returns_positive_reward_and_done(self):
        wrapper = _make_single_stage_wrapper()
        wrapper.listener.get_key.return_value = "c"
        intervened, done, reward = wrapper._check_keypress()
        assert reward == 1
        assert done is True
        assert intervened is True

    def test_no_key_does_not_print(self, capsys):
        wrapper = _make_single_stage_wrapper()
        wrapper.listener.get_key.return_value = None
        wrapper._check_keypress()
        assert capsys.readouterr().out == ""

    def test_key_pressed_prints_key(self, capsys):
        wrapper = _make_single_stage_wrapper()
        wrapper.listener.get_key.return_value = "c"
        wrapper._check_keypress()
        captured = capsys.readouterr().out
        assert "c" in captured


# ---------------------------------------------------------------------------
# KeyboardRewardDoneMultiStageWrapper — reward/done logic
# ---------------------------------------------------------------------------


class TestKeyboardRewardDoneMultiStageWrapper:
    def test_no_key_returns_stage0_reward(self):
        wrapper = _make_multi_stage_wrapper()
        wrapper.listener.get_key.return_value = None
        _intervened, done, reward = wrapper._check_keypress()
        assert reward == 0
        assert done is False

    def test_key_b_advances_to_stage1(self):
        wrapper = _make_multi_stage_wrapper()
        wrapper.listener.get_key.return_value = "b"
        _intervened, done, reward = wrapper._check_keypress()
        assert reward == pytest.approx(0.1)
        assert done is False

    def test_key_c_advances_to_stage2_and_done(self):
        wrapper = _make_multi_stage_wrapper()
        wrapper.listener.get_key.return_value = "c"
        _intervened, done, reward = wrapper._check_keypress()
        assert reward == pytest.approx(1.0)
        assert done is True

    def test_key_q_gives_penalty_not_done(self):
        wrapper = _make_multi_stage_wrapper()
        wrapper.reward_stage = 2
        wrapper.listener.get_key.return_value = "q"
        _intervened, done, reward = wrapper._check_keypress()
        assert reward == -1
        assert done is False

    def test_no_key_does_not_print(self, capsys):
        wrapper = _make_multi_stage_wrapper()
        wrapper.listener.get_key.return_value = None
        wrapper._check_keypress()
        assert capsys.readouterr().out == ""

    def test_key_pressed_prints_key(self, capsys):
        wrapper = _make_multi_stage_wrapper()
        wrapper.listener.get_key.return_value = "b"
        wrapper._check_keypress()
        captured = capsys.readouterr().out
        assert "b" in captured
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/keweijie/Desktop/gitcode/RLinf
python -m pytest tests/unit_tests/envs/realworld/test_keyboard_reward_done_wrapper.py -v
```

Expected: `test_no_key_does_not_print` fails in both classes because the current code always prints. All other tests should pass already.

---

### Task 2: Fix print spam in `reward_done_wrapper.py`

**Files:**
- Modify: `rlinf/envs/realworld/common/wrappers/reward_done_wrapper.py:57,90`

- [ ] **Step 1: Apply the fix**

In `KeyboardRewardDoneWrapper._check_keypress` (around line 57), change:

```python
        key = self.listener.get_key()
        print(f"Key pressed: {key}")
        if key not in ["a", "b", "c"]:
```

to:

```python
        key = self.listener.get_key()
        if key is not None:
            print(f"Key pressed: {key}")
        if key not in ["a", "b", "c"]:
```

In `KeyboardRewardDoneMultiStageWrapper._check_keypress` (around line 90), change:

```python
        key = self.listener.get_key()
        print(f"Key pressed: {key}")
        if key == "a":
```

to:

```python
        key = self.listener.get_key()
        if key is not None:
            print(f"Key pressed: {key}")
        if key == "a":
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python -m pytest tests/unit_tests/envs/realworld/test_keyboard_reward_done_wrapper.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add rlinf/envs/realworld/common/wrappers/reward_done_wrapper.py \
        tests/unit_tests/envs/realworld/__init__.py \
        tests/unit_tests/envs/realworld/test_keyboard_reward_done_wrapper.py
git commit -m "fix: only print key name when a key is actually pressed

Syncs reward_done_wrapper.py with RLinf_jie reference implementation.
Previously printed 'Key pressed: None' on every step when no key held."
```

---

### Task 3: Enable keyboard reward in Piper env config

**Files:**
- Modify: `examples/embodiment/config/env/realworld_piper_collect.yaml`

- [ ] **Step 1: Add `keyboard_reward_wrapper` field**

Open `examples/embodiment/config/env/realworld_piper_collect.yaml` and add after `use_spacemouse: False`:

```yaml
use_spacemouse: False
# keyboard_reward_wrapper: single_stage  # options: single_stage, multi_stage
# single_stage: A=-1(done) / B=0 / C=+1(done)
# multi_stage:  A/B/C set stage [0, 0.1, 1.0]; Q=-1 penalty
```

The field is commented out by default so existing training runs that reference this config are unaffected. Uncomment and set the desired mode to enable.

- [ ] **Step 2: Verify the YAML parses correctly**

```bash
python -c "
import yaml
with open('examples/embodiment/config/env/realworld_piper_collect.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg)
assert 'use_spacemouse' in cfg
print('YAML OK')
"
```

Expected output: dict printed with all keys, then `YAML OK`.

- [ ] **Step 3: Commit**

```bash
git add examples/embodiment/config/env/realworld_piper_collect.yaml
git commit -m "feat: add keyboard_reward_wrapper option to piper env config

Enables human-in-the-loop reward labeling for Piper online RL training.
Commented out by default to keep existing runs unaffected.
Set keyboard_reward_wrapper: single_stage or multi_stage to activate."
```

---

## Self-Review

**Spec coverage:**
- ✅ Fix print in `KeyboardRewardDoneWrapper._check_keypress` → Task 2 Step 1
- ✅ Fix print in `KeyboardRewardDoneMultiStageWrapper._check_keypress` → Task 2 Step 1
- ✅ Add `keyboard_reward_wrapper` to `realworld_piper_collect.yaml` → Task 3
- ✅ `keyboard_listener.py` unchanged (already identical to RLinf_jie) — no task needed

**Placeholder scan:** No TBD/TODO in any step. All code blocks complete.

**Type consistency:** `_check_keypress` returns `tuple[bool, bool, float]` throughout — consistent with base class signature and tests.
