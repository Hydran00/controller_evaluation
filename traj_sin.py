from initialize import get_robot_params
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import time
import matplotlib.pyplot as plt
import numpy as np
from plot import plot_trajectory


class SinusoidalTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("sinusoidal_trajectory_publisher")

        self.topic_name, self.base, self.end_effector = get_robot_params()
        # print namespace
        self.get_logger().info(f"Namespace: {self.get_namespace()}")
        self.output_txt_name = "outputs/sinusoidal_trajectory" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
        self.output_img_name = "outputs/sinusoidal_trajectory" + time.strftime("%Y%m%d-%H%M%S") + ".png"

        # Variables for the trajectory
        self.total_duration = 5.0  # Total duration for trajectory
        self.amplitude = 0.05  # Amplitude of the sinusoidal trajector

        self.axis_flag = [0, 0, 1.0]  # Which axis to move (x, y, z)

        # Sinusoidal parameters
        self.period = 5.0  # Period of the sinusoidal trajectory
        self.angular_frequency = 2 * np.pi / self.period  # Angular frequency

        # Initialize the publisher for PoseStamped messages
        self.publisher_ = self.create_publisher(PoseStamped, self.topic_name, 10)

        # Initialize tf2 for transforming coordinates
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Lists to store the commanded and executed trajectories for all axes
        self.commanded_trajectory_x = []
        self.commanded_trajectory_y = []
        self.commanded_trajectory_z = []
        self.executed_trajectory_x = []
        self.executed_trajectory_y = []
        self.executed_trajectory_z = []
        self.log_time = []


        # Variable to store current robot pose and orientation
        self.current_pose = None
        self.initial_orientation = None
        self.initial_position = None

        # Timer to periodically publish the trajectory (1 kHz)
        self.timer = self.create_timer(1e-3, self.publish_trajectory)  # 1 ms = 1 kHz
        self.cycle_iteration = 0

    def publish_trajectory(self):
        self.cycle_iteration += 1
        if self.cycle_iteration < 2000:
            self.start_time = time.time()
            return

        # Get the current time
        current_time = time.time()

        # Calculate the elapsed time since the start
        elapsed_time = current_time - self.start_time
        self.log_time.append(elapsed_time)

        if elapsed_time > self.total_duration:
            self.get_logger().info("Trajectory completed.")
            self.log_time = np.array(self.log_time)
            plot_trajectory(self)
            rclpy.shutdown()
            return

        # Get the current position and orientation of the robot's end effector (lbr_link_ee) in the base frame (lbr_link_0)
        try:
            # Get the transform between the end effector and the base link
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.base, self.end_effector, rclpy.time.Time()
            )

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
            self.get_logger().info(f"Current Pose: {self.current_pose}")
        except Exception as e:
            self.get_logger().warn(f"Could not get the current transform: {e}")
            return

        # Sinusoidal trajectory: x(t) = A * sin(ω * t)
        commanded_x = self.initial_position[0] + self.axis_flag[
            0
        ] * self.amplitude * np.sin(self.angular_frequency * elapsed_time)
        commanded_y = self.initial_position[1] + self.axis_flag[
            1
        ] * self.amplitude * np.sin(self.angular_frequency * elapsed_time)
        commanded_z = self.initial_position[2] + self.axis_flag[
            2
        ] * self.amplitude * np.sin(self.angular_frequency * elapsed_time)

        # Create PoseStamped message for commanded trajectory
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.base

        # Set commanded position
        pose.pose.position.x = commanded_x
        pose.pose.position.y = commanded_y
        pose.pose.position.z = commanded_z

        # Set orientation to the original orientation of the end effector
        pose.pose.orientation = self.initial_orientation

        self.get_logger().info(f"Target Pose: {pose}")

        # Publish the message
        self.publisher_.publish(pose)

        # Store the commanded and executed trajectories for later plotting
        self.commanded_trajectory_x.append(commanded_x)
        self.commanded_trajectory_y.append(commanded_y)
        self.commanded_trajectory_z.append(commanded_z)

        self.executed_trajectory_x.append(self.current_pose[0])
        self.executed_trajectory_y.append(self.current_pose[1])
        self.executed_trajectory_z.append(self.current_pose[2])


def main(args=None):
    rclpy.init(args=args)
    node = SinusoidalTrajectoryPublisher()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
