from initialize import get_robot_params
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformListener, Buffer
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation as R
import numpy as np
from pynput import keyboard
import signal
import sys
import time
from plot import plot_trajectory
from plot_cone_constraints import plot_cone_constraints


def quat_to_rpy(quat):
    """Converts a quaternion to roll-pitch-yaw angles."""
    r = R.from_quat(quat)
    return r.as_euler("xyz")


class KeyboardTeleop(Node):
    def signal_handler(self, sig, frame):
        print("plotting")
        plot_trajectory(self)
        sys.exit(0)

    def __init__(self):
        super().__init__("keyboard_control_trajectory_publisher")
        # capture SIGINT signal, e.g., Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        self.output_img_name = (
            "outputs/keyboard_teleop-" + time.strftime("%Y%m%d-%H%M%S") + ".png"
        )
        self.output_txt_name = (
            "outputs/keyboard_teleop-" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
        )
        self.log_time = [time.time()]
        self.plot_results = True
        self.total_duration = 4.0
        self.iter = 0
        self.xyz_increment = 0.01
        self.rpy_increment = np.radians(3)
        # Use simulation time
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        # Get robot parameters
        self.topic_name, self.base, self.end_effector = get_robot_params()

        # Publisher
        self.publisher_ = self.create_publisher(PoseStamped, self.topic_name, 10)

        # TF2 for transform lookup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_freq = 1000.0  # Hz

        # Initialize position and orientation
        self.target_position = [0.0, 0.0, 0.0]  # [x, y, z]
        self.target_orientation_rpy = [0.0, 0.0, 0.0]  # [roll, pitch, yaw]

        # Create a timer for trajectory publication
        self.timer = self.create_timer(1.0 / self.pub_freq, self.publish_trajectory)

        # Start listening to keyboard input
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

        # Inform the user of the controls
        self.get_logger().info(
            f"Initialized node with base: {self.base}, end_effector: {self.end_effector}, "
            f"publishing to topic: {self.topic_name}."
        )
        self.get_logger().info(
            "Use arrow keys for X/Y movement, 'w/s' for Z movement. "
            "'q/e' for roll, 'a/d' for pitch, 'z/c' for yaw."
        )
        self.pose_received = False
        self.commanded_trajectory_x = np.zeros(int(self.total_duration * self.pub_freq))
        self.commanded_trajectory_y = np.zeros(int(self.total_duration * self.pub_freq))
        self.commanded_trajectory_z = np.zeros(int(self.total_duration * self.pub_freq))
        self.commanded_trajectory_R = np.zeros(
            (int(self.total_duration * self.pub_freq), 3, 3)
        )
        self.executed_trajectory_x = np.zeros(int(self.total_duration * self.pub_freq))
        self.executed_trajectory_y = np.zeros(int(self.total_duration * self.pub_freq))
        self.executed_trajectory_z = np.zeros(int(self.total_duration * self.pub_freq))
        self.orientation_trajectory = np.zeros(
            (int(self.total_duration * self.pub_freq), 3, 3)
        )
        self.timesteps = np.zeros(int(self.total_duration * self.pub_freq))

    def on_key_press(self, key):
        """Updates position and orientation based on keyboard input."""
        if self.pose_received is False:
            self.get_logger().warn("No transform received yet. Cannot update pose.")
            return
        try:
            if hasattr(key, "char") and key.char:
                if key.char == "w":
                    self.target_position[2] += self.xyz_increment  # Z-axis up
                elif key.char == "s":
                    self.target_position[2] -= self.xyz_increment  # Z-axis down
                elif key.char == "q":
                    self.target_orientation_rpy[0] += self.rpy_increment  # Roll
                elif key.char == "e":
                    self.target_orientation_rpy[0] -= self.rpy_increment  # Roll
                elif key.char == "a":
                    self.target_orientation_rpy[1] += self.rpy_increment  # Pitch
                elif key.char == "d":
                    self.target_orientation_rpy[1] -= self.rpy_increment  # Pitch
                elif key.char == "z":
                    self.target_orientation_rpy[2] += self.rpy_increment  # Yaw
                elif key.char == "c":
                    self.target_orientation_rpy[2] -= self.rpy_increment  # Yaw
            elif hasattr(key, "name"):
                if key.name == "up":
                    self.target_position[1] += self.xyz_increment  # Y-axis forward
                elif key.name == "down":
                    self.target_position[1] -= self.xyz_increment  # Y-axis backward
                elif key.name == "right":
                    self.target_position[0] += self.xyz_increment  # X-axis right
                elif key.name == "left":
                    self.target_position[0] -= self.xyz_increment  # X-axis left

            self.target_orientation_rpy[0] %= 2 * np.pi
            self.target_orientation_rpy[1] %= 2 * np.pi
            self.target_orientation_rpy[2] %= 2 * np.pi
        except Exception as e:
            self.get_logger().warn(f"Error processing key press: {e}")

    def publish_trajectory(self):
        """Publishes the updated PoseStamped message."""
        self.current_transform = self.get_current_transform()
        if self.current_transform is None:
            return
        self.current_time = time.time() - self.start_time

        # Create PoseStamped message
        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = self.base

        # Set position
        target_pose.pose.position.x = self.target_position[0]
        target_pose.pose.position.y = self.target_position[1]
        target_pose.pose.position.z = self.target_position[2]

        # Convert RPY to quaternion
        quat = R.from_euler("xyz", self.target_orientation_rpy).as_quat()
        # Set orientation
        target_pose.pose.orientation.x = quat[0]
        target_pose.pose.orientation.y = quat[1]
        target_pose.pose.orientation.z = quat[2]
        target_pose.pose.orientation.w = quat[3]

        # Publish the message
        self.publisher_.publish(target_pose)
        self.get_logger().info(
            f"Published pose -> Position: {self.target_position}, Orientation (RPY): {np.degrees(self.target_orientation_rpy)}"
        )
        # Store the commanded and executed trajectories for later plotting
        if self.plot_results and self.iter >= self.total_duration * self.pub_freq:
            plot_trajectory(self)
            init_orientation_matrix = R.from_quat(self.initial_orientation).as_matrix()
            reference_orientation_matrix = R.from_quat(
                self.initial_orientation
            ).as_matrix()
            plot_cone_constraints(
                self.timesteps,
                reference_orientation_matrix,
                self.orientation_trajectory,
                None,
                self.commanded_trajectory_R,
                [0.4, 0.4, 0.4],
            )

            sys.exit(0)
        self.commanded_trajectory_x[self.iter] = target_pose.pose.position.x
        self.commanded_trajectory_y[self.iter] = target_pose.pose.position.y
        self.commanded_trajectory_z[self.iter] = target_pose.pose.position.z

        self.executed_trajectory_x[self.iter] = (
            self.current_transform.transform.translation.x
        )
        self.executed_trajectory_y[self.iter] = (
            self.current_transform.transform.translation.y
        )
        self.executed_trajectory_z[self.iter] = (
            self.current_transform.transform.translation.z
        )
        self.orientation_trajectory[self.iter] = R.from_quat(
            [
                self.current_transform.transform.rotation.x,
                self.current_transform.transform.rotation.y,
                self.current_transform.transform.rotation.z,
                self.current_transform.transform.rotation.w,
            ]
        ).as_matrix()
        self.commanded_trajectory_R[self.iter] = R.from_quat(quat).as_matrix()
        self.timesteps[self.iter] = self.current_time
        self.iter += 1
        self.log_time.append(time.time())

    def get_current_transform(self):
        """Retrieve the current transform between base and end-effector."""
        try:
            transform_msg: TransformStamped = self.tf_buffer.lookup_transform(
                self.base, self.end_effector, rclpy.time.Time()
            )
            if not self.pose_received:
                self.get_logger().info(f"Got initial transform: {transform_msg}")
                self.target_position = [
                    transform_msg.transform.translation.x,
                    transform_msg.transform.translation.y,
                    transform_msg.transform.translation.z,
                ]
                self.initial_orientation = [
                    transform_msg.transform.rotation.x,
                    transform_msg.transform.rotation.y,
                    transform_msg.transform.rotation.z,
                    transform_msg.transform.rotation.w,
                ]
                self.target_orientation_rpy = quat_to_rpy(self.initial_orientation)
                self.start_time = time.time()
                self.pose_received = True
            return transform_msg
        except Exception as e:
            self.get_logger().warn(f"Failed to get transform: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
