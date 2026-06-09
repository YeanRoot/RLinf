#!/usr/bin/env python3
"""Small Piper gripper probe used by Codex during real-robot debugging."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import rospy
from sensor_msgs.msg import JointState


@dataclass
class ArmState:
    position: np.ndarray | None = None
    effort: np.ndarray | None = None
    stamp: float = 0.0


def _joint_callback(state: ArmState):
    def _callback(msg: JointState) -> None:
        if len(msg.position) >= 7:
            state.position = np.asarray(msg.position[:7], dtype=np.float64)
        if len(msg.effort) >= 7:
            state.effort = np.asarray(msg.effort[:7], dtype=np.float64)
        state.stamp = time.time()

    return _callback


def _wait_for_state(state: ArmState, name: str, timeout: float) -> np.ndarray:
    start = time.time()
    while state.position is None:
        if time.time() - start > timeout:
            raise RuntimeError(f"Timed out waiting for {name} joint state")
        time.sleep(0.02)
    return state.position.copy()


def _make_joint_msg(position: np.ndarray, speed: float, effort: float) -> JointState:
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    msg.position = np.asarray(position, dtype=np.float64).tolist()
    msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(speed)]
    msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(effort)]
    return msg


def _fmt(x: np.ndarray | None) -> str:
    if x is None:
        return "None"
    return "[" + ", ".join(f"{v:+.6f}" for v in x.tolist()) + "]"


def command_gripper(args: argparse.Namespace) -> None:
    rospy.init_node("codex_gripper_probe", anonymous=True)

    left_state = ArmState()
    right_state = ArmState()
    rospy.Subscriber(args.puppet_left_topic, JointState, _joint_callback(left_state), queue_size=1)
    rospy.Subscriber(args.puppet_right_topic, JointState, _joint_callback(right_state), queue_size=1)

    left_pub = rospy.Publisher(args.master_left_topic, JointState, queue_size=1)
    right_pub = rospy.Publisher(args.master_right_topic, JointState, queue_size=1)

    left_pos = _wait_for_state(left_state, "left puppet", args.timeout)
    right_pos = _wait_for_state(right_state, "right puppet", args.timeout)
    before_left = left_pos.copy()
    before_right = right_pos.copy()

    if args.disable_teleop_param:
        rospy.set_param("/enable_message_publish", False)
        time.sleep(0.1)

    target_left = before_left.copy()
    target_right = before_right.copy()
    if args.side in ("left", "both"):
        target_left[6] = args.gripper
    if args.side in ("right", "both"):
        target_right[6] = args.gripper

    print(f"teleop_param_before={rospy.get_param('/enable_message_publish', None)}", flush=True)
    print(f"before_left={_fmt(before_left)} effort={_fmt(left_state.effort)}", flush=True)
    print(f"before_right={_fmt(before_right)} effort={_fmt(right_state.effort)}", flush=True)
    print(
        f"command side={args.side} gripper={args.gripper:.6f} "
        f"duration={args.duration:.3f}s rate={args.rate:.1f}Hz speed={args.speed:.1f} effort={args.effort:.3f}",
        flush=True,
    )

    rate = rospy.Rate(args.rate)
    end_time = time.time() + args.duration
    while time.time() < end_time and not rospy.is_shutdown():
        if args.side in ("left", "both"):
            left_pub.publish(_make_joint_msg(target_left, args.speed, args.effort))
        if args.side in ("right", "both"):
            right_pub.publish(_make_joint_msg(target_right, args.speed, args.effort))
        rate.sleep()

    time.sleep(args.settle)
    after_left = left_state.position.copy() if left_state.position is not None else None
    after_right = right_state.position.copy() if right_state.position is not None else None
    print(f"teleop_param_after={rospy.get_param('/enable_message_publish', None)}", flush=True)
    print(f"after_left={_fmt(after_left)} effort={_fmt(left_state.effort)}", flush=True)
    print(f"after_right={_fmt(after_right)} effort={_fmt(right_state.effort)}", flush=True)


def monitor(args: argparse.Namespace) -> None:
    rospy.init_node("codex_gripper_monitor", anonymous=True)
    topics = {
        "puppet_left": args.puppet_left_topic,
        "puppet_right": args.puppet_right_topic,
        "master_left_state": args.master_left_state_topic,
        "master_right_state": args.master_right_state_topic,
        "master_left_cmd": args.master_left_topic,
        "master_right_cmd": args.master_right_topic,
    }
    states = {name: ArmState() for name in topics}
    for name, topic in topics.items():
        rospy.Subscriber(topic, JointState, _joint_callback(states[name]), queue_size=1)

    start = time.time()
    next_print = 0.0
    while time.time() - start < args.duration and not rospy.is_shutdown():
        now = time.time()
        if now >= next_print:
            vals = []
            for name in topics:
                pos = states[name].position
                effort = states[name].effort
                g = None if pos is None else pos[6]
                e = None if effort is None else effort[6]
                vals.append(
                    f"{name}.g={g:+.6f}" if g is not None else f"{name}.g=None"
                )
                vals.append(
                    f"{name}.eff={e:+.4f}" if e is not None else f"{name}.eff=None"
                )
            vals.append(f"teleop={rospy.get_param('/enable_message_publish', None)}")
            print(" | ".join(vals), flush=True)
            next_print = now + args.interval
        time.sleep(0.02)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["command", "monitor"], default="command")
    parser.add_argument("--side", choices=["left", "right", "both"], default="right")
    parser.add_argument("--gripper", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--effort", type=float, default=1.0)
    parser.add_argument("--settle", type=float, default=0.3)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--disable-teleop-param", action="store_true")
    parser.add_argument("--puppet-left-topic", default="/puppet/joint_left")
    parser.add_argument("--puppet-right-topic", default="/puppet/joint_right")
    parser.add_argument("--master-left-state-topic", default="/puppet/joint_l_master")
    parser.add_argument("--master-right-state-topic", default="/puppet/joint_r_master")
    parser.add_argument("--master-left-topic", default="/master/joint_left")
    parser.add_argument("--master-right-topic", default="/master/joint_right")
    args = parser.parse_args()

    if args.mode == "monitor":
        monitor(args)
    else:
        command_gripper(args)


if __name__ == "__main__":
    main()
