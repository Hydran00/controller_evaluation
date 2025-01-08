from initialize import get_robot_params

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import time
import numpy as np
from plot import plot_trajectory
import time


class LinTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("linear_trajectory_publisher")
        # set use_sim_time to True
        self.use_sim_time = True
        self.topic_name, self.base, self.end_effector = get_robot_params()
        # Initialize the publisher for PoseStamped messages
        self.publisher_ = self.create_publisher(PoseStamped, self.topic_name, 10)

        # Initialize tf2 for transforming coordinates
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_freq = 1000  # 1 kHz
        self.settling_time = 2.0  # initial and final buffer time

        # Variables for the trajectory
        self.total_duration = 8.0  # Total duration for trajectory
        self.delta = -0.2  # Speed of the linear trajectory

        # Lists to store the commanded and executed trajectories for all axes
        self.commanded_trajectory_x = []
        self.commanded_trajectory_y = []
        self.commanded_trajectory_z = []
        self.executed_trajectory_x = []
        self.executed_trajectory_y = []
        self.executed_trajectory_z = []

        # Variable to store current robot pose and orientation
        self.current_pose = None
        self.initial_orientation = None
        self.initial_position = None
        self.cycle_iteration = 0
        self.move_target = False
        self.last_cycle_check = 0

        self.start_time = time.time()
        self.get_logger().warn(
            f"Base is {self.base} and end effector is {self.end_effector}"
        )
        self.get_logger().warn(f"Topic name is {self.topic_name}")
        time.sleep(1.5)
        self.state = "initial_waiting"
        # Timer to periodically publish the trajectory (1 kHz)
        self.timer = self.create_timer(
            1.0 / self.pub_freq, self.publish_trajectory
        )  # 1 ms = 1 kHz

    def get_current_transform(self):
        try:
            transform_msg: TransformStamped = self.tf_buffer.lookup_transform(
                self.base, self.end_effector, rclpy.time.Time()
            )
            return transform_msg
        except Exception as e:
            self.get_logger().warn(f"Failed to get transform: {e}")
            return None

    def publish_trajectory(self):
        self.cycle_iteration += 1

        transform_msg = self.get_current_transform()
        if transform_msg is None:
            return

        print(f"Cycle iteration: {self.cycle_iteration}")
        if self.state == "initial_waiting":
            self.get_logger().info("Initial waiting...")
            # Set the initial orientation and position
            if self.initial_orientation is None:
                self.initial_orientation = transform_msg.transform.rotation
                self.initial_position = (
                    transform_msg.transform.translation.x,
                    transform_msg.transform.translation.y,
                    transform_msg.transform.translation.z,
                )
                self.commanded_x = self.initial_position[0]
                self.commanded_y = self.initial_position[1]
                self.commanded_z = self.initial_position[2]

            if self.cycle_iteration == int(self.settling_time * self.pub_freq):
                self.start_time = time.time()
                self.state = "moving"
                self.move_target = True

        if self.state == "moving":
            self.get_logger().info("Moving...")
            # Get the current time
            current_time = time.time()

            # Calculate the elapsed time since the start
            elapsed_time = current_time - self.start_time
            if elapsed_time > self.total_duration:
                self.move_target = False
                self.state = "final_waiting"

        if self.state == "final_waiting":
            self.get_logger().info("Final waiting...")
            if self.last_cycle_check == int(self.settling_time * self.pub_freq):
                self.get_logger().info("Trajectory completed")
                plot_trajectory(self)
                rclpy.shutdown()
            self.last_cycle_check += 1

        self.current_pose = (
            transform_msg.transform.translation.x,
            transform_msg.transform.translation.y,
            transform_msg.transform.translation.z,
        )

        # self.get_logger().info(f"Current Pose: {self.current_pose}")

        if self.move_target:
            # Update the commanded trajectory based on the elapsed time
            # Linear trajectory: y(t) = A * t
            self.commanded_x = self.initial_position[0]
            self.commanded_y = self.initial_position[1]  # + self.delta * elapsed_time
            self.commanded_z = self.initial_position[2]  # + self.delta * elapsed_time

        self.get_logger().info(
            f"Commanded Pose: ({self.commanded_x}, {self.commanded_y}, {self.commanded_z})"
        )
        # Create PoseStamped message for commanded trajectory
        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = self.base
        # Set commanded position
        target_pose.pose.position.x = self.commanded_x
        target_pose.pose.position.y = self.commanded_y
        target_pose.pose.position.z = self.commanded_z

        # Set orientation to the original orientation of the end effector
        target_pose.pose.orientation = self.initial_orientation

        # Publish the message
        self.publisher_.publish(target_pose)

        # Store the commanded and executed trajectories for later plotting
        self.commanded_trajectory_x.append(self.commanded_x)
        self.commanded_trajectory_y.append(self.commanded_y)
        self.commanded_trajectory_z.append(self.commanded_z)

        self.executed_trajectory_x.append(self.current_pose[0])
        self.executed_trajectory_y.append(self.current_pose[1])
        self.executed_trajectory_z.append(self.current_pose[2])


def main(args=None):
    rclpy.init(args=args)
    node = LinTrajectoryPublisher()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
