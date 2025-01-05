from initialize import get_robot_params

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import time
import matplotlib.pyplot as plt
import numpy as np


class LinTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("homing_cart_trajectory_publisher")

        self.topic_name, self.base, self.end_effector = get_robot_params()
        # Initialize the publisher for PoseStamped messages
        self.publisher_ = self.create_publisher(PoseStamped, self.topic_name, 10)

        # Initialize tf2 for transforming coordinates
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Variables for the trajectory
        self.target_position = (0.5, 0.3, 0.6)  # Target position for the end effector
        self.target_orientation = (
            1.0,
            0.0,
            0.0,
            0.0,
        )  # Target orientation for the end effector
        self.total_duration = 0.0
        self.max_speed = 0.05  # m/s

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

        self.start_time = time.time()
        self.last_measurement_time = time.time()
        self.get_logger().warn(
            f"Base is {self.base} and end effector is {self.end_effector}"
        )

        # Timer to periodically publish the trajectory (1 kHz)
        self.timer = self.create_timer(1e-3, self.publish_trajectory)  # 1 ms = 1 kHz
        self.iter = 0
        self.step = 0

    def publish_trajectory(self):
        while self.iter < 2000:
            self.start_time = time.time()
            self.iter += 1
            return
        # Get the current time
        current_time = time.time()

        # Calculate the elapsed time since the start
        elapsed_time = current_time - self.start_time
        deltaT = current_time - self.last_measurement_time

        # Get the current position and orientation of the robot's end effector in the base frame
        try:
            # Get the transform between the end effector and the base link
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.base, self.end_effector, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"Could not get the current transform: {e}")
            return

        # Extract the translation (position) and orientation of the end effector
        current_x = transform.transform.translation.x
        current_y = transform.transform.translation.y
        current_z = transform.transform.translation.z
        current_orientation = transform.transform.rotation

        # Set the initial orientation if not already set
        if self.initial_orientation is None:
            self.initial_orientation = current_orientation
            self.initial_position = (current_x, current_y, current_z)

        self.current_pose = (current_x, current_y, current_z)
        if (
            np.linalg.norm(np.array(self.current_pose) - np.array(self.target_position))
            < 0.2
        ):
            self.get_logger().info("Trajectory completed.")
            self.total_duration = elapsed_time
            self.plot_trajectory()
            rclpy.shutdown()
            return

        if self.iter == 2000:
            self.distance = np.linalg.norm(
                np.array(self.current_pose) - np.array(self.target_position)
            )
            self.direction = (
                np.array(self.target_position) - np.array(self.current_pose)
            ) / np.linalg.norm(
                np.array(self.target_position) - np.array(self.current_pose)
            )
            self.speed = self.max_speed * self.direction

        # Linear trajectory: y(t) = A * t
        commanded_x = self.initial_position[0] + self.speed[0] * deltaT * self.step
        commanded_y = self.initial_position[1] + self.speed[1] * deltaT * self.step
        commanded_z = self.initial_position[2] + self.speed[2] * deltaT * self.step
        self.step += 1

        self.get_logger().warning(
            f"Distance Remaining: {np.linalg.norm(np.array(self.current_pose) - np.array(self.target_position))}"
        )
        # Create PoseStamped message for commanded trajectory
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.base
        # Set commanded position
        pose.pose.position.x = commanded_x
        pose.pose.position.y = commanded_y
        pose.pose.position.z = commanded_z

        # Set orientation to the original orientation of the end effector
        pose.pose.orientation.x = self.target_orientation[0]
        pose.pose.orientation.y = self.target_orientation[1]
        pose.pose.orientation.z = self.target_orientation[2]
        pose.pose.orientation.w = self.target_orientation[3]

        # Publish the message
        self.publisher_.publish(pose)

        # Store the commanded and executed trajectories for later plotting
        self.commanded_trajectory_x.append(commanded_x)
        self.commanded_trajectory_y.append(commanded_y)
        self.commanded_trajectory_z.append(commanded_z)

        self.executed_trajectory_x.append(self.current_pose[0])
        self.executed_trajectory_y.append(self.current_pose[1])
        self.executed_trajectory_z.append(self.current_pose[2])

        self.last_measurement_time = current_time

    def plot_trajectory(self):
        # Plot the commanded and executed trajectories for all axes
        time_steps = np.linspace(
            0, self.total_duration, len(self.commanded_trajectory_x)
        )

        plt.figure(figsize=(12, 8))

        # Plot x-axis trajectories
        plt.subplot(3, 1, 1)
        plt.plot(
            time_steps,
            self.commanded_trajectory_x,
            label="Commanded X",
            linestyle="-",
            color="b",
        )
        plt.plot(
            time_steps,
            self.executed_trajectory_x,
            label="Executed X",
            linestyle="-",
            color="r",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.title("X-Axis: Commanded vs Executed")
        plt.legend()
        plt.grid(True)

        # Plot y-axis trajectories
        plt.subplot(3, 1, 2)
        plt.plot(
            time_steps,
            self.commanded_trajectory_y,
            label="Commanded Y",
            linestyle="-",
            color="b",
        )
        plt.plot(
            time_steps,
            self.executed_trajectory_y,
            label="Executed Y",
            linestyle="-",
            color="r",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.title("Y-Axis: Commanded vs Executed")
        plt.legend()
        plt.grid(True)

        # Plot z-axis trajectories
        plt.subplot(3, 1, 3)
        plt.plot(
            time_steps,
            self.commanded_trajectory_z,
            label="Commanded Z",
            linestyle="-",
            color="b",
        )
        plt.plot(
            time_steps,
            self.executed_trajectory_z,
            label="Executed Z",
            linestyle="-",
            color="r",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.title("Z-Axis: Commanded vs Executed")
        plt.legend()
        plt.grid(True)

        # Show all plots
        plt.tight_layout()

        # Plot the error between the commanded and executed trajectories
        plt.figure(figsize=(12, 8))

        # Plot x-axis error
        plt.subplot(3, 1, 1)
        error_x = np.array(self.executed_trajectory_x) - np.array(
            self.commanded_trajectory_x
        )
        plt.plot(time_steps, error_x, label="Error X", linestyle="-", color="g")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.title("X-Axis Error")
        plt.legend()
        plt.grid(True)

        # Plot y-axis error
        plt.subplot(3, 1, 2)
        error_y = np.array(self.executed_trajectory_y) - np.array(
            self.commanded_trajectory_y
        )
        plt.plot(time_steps, error_y, label="Error Y", linestyle="-", color="g")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.title("Y-Axis Error")
        plt.legend()
        plt.grid(True)

        # Plot z-axis error
        plt.subplot(3, 1, 3)
        error_z = np.array(self.executed_trajectory_z) - np.array(
            self.commanded_trajectory_z
        )
        plt.plot(time_steps, error_z, label="Error Z", linestyle="-", color="g")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.title("Z-Axis Error")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Calculate and print the mean square error for each direction
        mse_x = np.mean(np.square(error_x))
        mse_y = np.mean(np.square(error_y))
        mse_z = np.mean(np.square(error_z))

        self.get_logger().info(
            f"Mean Square Error - X: {mse_x}, Y: {mse_y}, Z: {mse_z}"
        )
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = LinTrajectoryPublisher()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
