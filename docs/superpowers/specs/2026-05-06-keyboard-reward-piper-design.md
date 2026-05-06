# Keyboard Reward for Piper Online RL Training

**Date:** 2026-05-06  
**Reference:** RLinf_jie (verified implementation)

## Goal

Enable human-in-the-loop reward labeling via keyboard during online RL training on the Piper robot arm, and sync `reward_done_wrapper.py` with the verified RLinf_jie implementation.

## Background

The `KeyboardListener` class and `KeyboardRewardDoneWrapper` / `KeyboardRewardDoneMultiStageWrapper` wrappers already exist. `realworld_env.py` already routes `keyboard_reward_wrapper: single_stage|multi_stage` config to the correct wrapper for both franka and piper_joint stacks. The Franka config (`realworld_bin_relocation.yaml`) already enables this feature. The Piper config does not.

## Key Mapping

| Key | single_stage | multi_stage |
|-----|-------------|-------------|
| A   | reward=-1, done=True | stage=0, reward=0 |
| B   | reward=0 | stage=1, reward=0.1 |
| C   | reward=+1, done=True | stage=2, reward=1.0, done=True |
| Q   | — | reward=-1 (penalty, no done) |

`reward_mode: always_replace` — every step's reward is replaced by keyboard reward (0 when no key is held).

## Changes

### 1. Fix print spam in `reward_done_wrapper.py`

Sync with RLinf_jie: only print when a key is actually pressed (not every step).

**File:** `rlinf/envs/realworld/common/wrappers/reward_done_wrapper.py`

- `KeyboardRewardDoneWrapper._check_keypress` line 57: wrap `print` with `if key is not None`
- `KeyboardRewardDoneMultiStageWrapper._check_keypress` line 90: same fix

### 2. Enable keyboard reward in Piper env config

**File:** `examples/embodiment/config/env/realworld_piper_collect.yaml`

Add `keyboard_reward_wrapper: single_stage` with a comment indicating `multi_stage` is also available.

## What is NOT changed

- `keyboard_listener.py` — identical between RLinf and RLinf_jie, no changes needed
- `realworld_env.py` — already supports piper_joint keyboard reward routing
- `last_intervened` logic in `KeyboardRewardDoneMultiStageWrapper` — with `always_replace` mode this flag is irrelevant; no change to avoid diverging from RLinf_jie
