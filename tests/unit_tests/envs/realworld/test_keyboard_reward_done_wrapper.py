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
