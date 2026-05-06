# Copyright 2025 The RLinf Authors.
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

"""Test script for Piper camera setup.

This script tests the multi-camera setup for Piper robot:
- 2 RealSense cameras (camera_l, camera_r)
- 1 Dabai (Orbbec Astra) camera (camera_f)

Usage:
    python test_piper_camera.py
"""

import time

import pyrealsense2 as rs


def test_realsense_cameras():
    """Test RealSense cameras via ROS topics.
    
    Note: When ROS realsense2_camera node is running, the cameras are occupied
    and cannot be accessed directly via pyrealsense2 SDK.
    So we test via ROS topics instead.
    """
    print("=" * 60)
    print("Testing RealSense Cameras (via ROS topics)")
    print("=" * 60)

    try:
        import rospy
        from sensor_msgs.msg import Image
        
        # Check if ROS node is already initialized
        if not rospy.core.is_initialized():
            rospy.init_node("test_realsense_cameras", anonymous=True)
        
        # Expected topics for camera_l and camera_r
        cameras = {
            "camera_l": {
                "color": "/camera_l/color/image_raw",
                "depth": "/camera_l/depth/image_rect_raw",
            },
            "camera_r": {
                "color": "/camera_r/color/image_raw",
                "depth": "/camera_r/depth/image_rect_raw",
            },
        }
        
        # Get all available topics
        all_topics = rospy.get_published_topics()
        available_topic_names = [topic[0] for topic in all_topics]
        
        print("\nChecking ROS topics for RealSense cameras...")
        
        all_ok = True
        for cam_name, topics in cameras.items():
            print(f"\n{cam_name}:")
            for stream_type, topic in topics.items():
                if topic in available_topic_names:
                    print(f"  ✓ {topic} available")
                else:
                    print(f"  ✗ {topic} NOT available")
                    all_ok = False
        
        if not all_ok:
            print("\n✗ Some RealSense topics not available.")
            print("  Make sure ROS camera nodes are running:")
            print("    roslaunch realsense2_camera multi_camera.launch")
            return False
        
        # Test receiving frames from each camera
        print("\n" + "-" * 60)
        print("Testing frame reception from RealSense cameras...")
        print("-" * 60)
        
        frame_received = {}
        
        def make_callback(cam_name, stream_type):
            def callback(msg):
                key = f"{cam_name}_{stream_type}"
                if key not in frame_received:
                    frame_received[key] = True
                    print(f"  ✓ {cam_name} {stream_type}: {msg.width}x{msg.height}")
            return callback
        
        subscribers = []
        for cam_name, topics in cameras.items():
            for stream_type, topic in topics.items():
                sub = rospy.Subscriber(
                    topic, Image, make_callback(cam_name, stream_type)
                )
                subscribers.append(sub)
        
        # Wait for frames
        timeout = 5.0
        start_time = time.time()
        expected_keys = [f"{cam}_{stream}" 
                        for cam in cameras 
                        for stream in cameras[cam]]
        
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() - start_time < timeout:
            if all(key in frame_received for key in expected_keys):
                print("\n✓ All RealSense cameras streaming OK!")
                return True
            rate.sleep()
        
        # Check what's missing
        missing = [key for key in expected_keys if key not in frame_received]
        if missing:
            print(f"\n✗ Timeout waiting for: {missing}")
            return False
        
        return True
        
    except ImportError:
        print("\n⚠ ROS Python not available")
        print("  Falling back to direct pyrealsense2 SDK test...")
        return test_realsense_cameras_direct()
    except Exception as e:
        print(f"\n✗ RealSense test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_realsense_cameras_direct():
    """Test RealSense cameras directly via pyrealsense2 SDK.
    
    Note: This only works when cameras are NOT occupied by ROS nodes.
    """
    print("\nTesting RealSense via pyrealsense2 SDK (direct access)...")
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("ERROR: No RealSense devices found!")
        print("  Note: If ROS realsense2_camera node is running,")
        print("  the cameras are occupied and cannot be accessed directly.")
        print("  Please use ROS topic test instead.")
        return False
    
    print(f"\nFound {len(devices)} RealSense device(s):\n")
    
    for i, device in enumerate(devices):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"  Device {i + 1}: {name} (SN: {serial})")
    
    return True


def test_dabai_camera():
    """Test Dabai (Orbbec Astra) camera via ROS topics.
    
    Note: This requires ROS to be running with astra_camera node.
    For standalone test, you would need the Orbbec SDK.
    """
    print("\n" + "=" * 60)
    print("Testing Dabai (Orbbec Astra) Camera")
    print("=" * 60)
    
    try:
        import rospy
        from sensor_msgs.msg import Image
        
        print("\nChecking ROS topics for Dabai camera...")

        
        # Initialize ROS node if not already initialized
        if not rospy.core.is_initialized():
            rospy.init_node("test_dabai_camera", anonymous=True)
        
        # Expected topics from multi_camera.launch
        expected_topics = [
            "/camera_f/color/image_raw",
            "/camera_f/depth/image_raw",
        ]
        
        # Get all available topics
        all_topics = rospy.get_published_topics()
        available_topic_names = [topic[0] for topic in all_topics]
        
        print("\nChecking expected topics:")
        for topic in expected_topics:
            if topic in available_topic_names:
                print(f"  ✓ {topic} available")
            else:
                print(f"  ✗ {topic} NOT available")
        
        # Try to subscribe and receive a frame
        print("\nTrying to receive frames from camera_f...")
        
        frame_received = {"color": False, "depth": False}
        
        def color_callback(msg):
            frame_received["color"] = True
            print(f"  ✓ Received color frame: {msg.width}x{msg.height}")
        
        def depth_callback(msg):
            frame_received["depth"] = True
            print(f"  ✓ Received depth frame: {msg.width}x{msg.height}")
        
        color_sub = rospy.Subscriber(
            "/camera_f/color/image_raw", Image, color_callback
        )
        depth_sub = rospy.Subscriber(
            "/camera_f/depth/image_raw", Image, depth_callback
        )
        
        # Wait for frames
        timeout = 5.0
        start_time = time.time()
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown() and time.time() - start_time < timeout:
            if frame_received["color"] and frame_received["depth"]:
                print("\n✓ Dabai camera test passed!")
                return True
            rate.sleep()
        
        if not (frame_received["color"] and frame_received["depth"]):
            print("\n✗ Timeout waiting for Dabai camera frames")
            return False
        
    except ImportError:
        print("\n⚠ ROS Python not available, skipping Dabai test")
        print("  To test Dabai camera, run:")
        print("    roslaunch realsense2_camera multi_camera.launch")
        print("  Then check topics with: rostopic list | grep camera_f")
        return None
    except Exception as e:
        print(f"\n✗ Dabai camera test failed: {e}")
        return False


def main():
    """Main test function."""
    print("\n" + "=" * 60)
    print("Piper Multi-Camera Test Suite")
    print("=" * 60)
    print("\nThis script tests the camera setup for Piper robot:")
    print("  - 2 x RealSense D435I cameras (camera_l, camera_r)")
    print("  - 1 x Dabai (Orbbec Astra) camera (camera_f)")
    print()
    
    # Test RealSense cameras
    realsense_ok = test_realsense_cameras()
    
    # Test Dabai camera (requires ROS)
    dabai_ok = test_dabai_camera()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"RealSense cameras: {'✓ PASS' if realsense_ok else '✗ FAIL'}")
    if dabai_ok is not None:
        print(f"Dabai camera:      {'✓ PASS' if dabai_ok else '✗ FAIL'}")
    else:
        print(f"Dabai camera:      ⚠ SKIPPED (ROS not available)")
    print()
    
    if realsense_ok and (dabai_ok is None or dabai_ok):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
