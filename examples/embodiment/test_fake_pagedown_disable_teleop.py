#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import time

import rospy


def as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ["true", "1", "yes", "y", "on"]
    return bool(x)


def fake_pagedown_by_xdotool():
    if shutil.which("xdotool") is None:
        return False, "xdotool not found"

    display = os.environ.get("DISPLAY")
    if not display:
        return False, "DISPLAY is not set"

    try:
        subprocess.run(
            ["xdotool", "keydown", "Page_Down"],
            check=True,
            timeout=1.0,
        )
        time.sleep(0.08)
        subprocess.run(
            ["xdotool", "keyup", "Page_Down"],
            check=True,
            timeout=1.0,
        )
        return True, "sent Page_Down by xdotool"
    except Exception as e:
        return False, f"xdotool failed: {repr(e)}"


def fake_pagedown_by_pynput():
    try:
        from pynput.keyboard import Controller, Key

        keyboard = Controller()
        keyboard.press(Key.page_down)
        time.sleep(0.08)
        keyboard.release(Key.page_down)
        return True, "sent PageDown by pynput"
    except Exception as e:
        return False, f"pynput failed: {repr(e)}"


def main():
    rospy.init_node("test_fake_pagedown_disable_teleop", anonymous=True)

    param_name = "/enable_message_publish"

    before_raw = rospy.get_param(param_name, False)
    before = as_bool(before_raw)

    print(f"[test] {param_name} before = {before_raw!r} -> {before}")

    if not before:
        print("[test] teleop is already disabled. Do nothing to avoid toggling it ON.")
        return

    print("[test] teleop is active. Try to fake one PageDown to toggle it OFF.")

    ok, msg = fake_pagedown_by_xdotool()
    print(f"[test] xdotool result: ok={ok}, msg={msg}")

    if not ok:
        ok, msg = fake_pagedown_by_pynput()
        print(f"[test] pynput result: ok={ok}, msg={msg}")

    time.sleep(0.3)

    after_raw = rospy.get_param(param_name, False)
    after = as_bool(after_raw)

    print(f"[test] {param_name} after  = {after_raw!r} -> {after}")

    if after:
        print(
            "[test][WARN] param is still True. "
            "The fake PageDown probably did not reach the teleop listener."
        )
    else:
        print(
            "[test][OK] param is now False. "
            "Now check whether the slave arm really stopped following the master."
        )


if __name__ == "__main__":
    main()