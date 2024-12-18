import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import time
import matplotlib.pyplot as plt
import numpy as np

ROBOT_TYPE = "franka" # "franka" or "kuka"

class LinTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('linear_trajectory_publisher')
        
        self.topic_name = "/cartesian_impedance_controller/target_frame"
        if ROBOT_TYPE == "franka":
            self.base = "base"
            self.end_effector = "fr3_hand_tcp"
        elif ROBOT_TYPE == "kuka":
            self.base = "lbr_link_0"
            self.end_effector = "lbr_link_ee"
        else:
            print("Robot type unknown")
            

        # Initialize the publisher for PoseStamped messages
        self.publisher_ = self.create_publisher(PoseStamped, 'target_frame', 10)

        # Initialize tf2 for transforming coordinates
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Variables for the trajectory
        self.total_duration = 3.0  # Total duration for trajectory
        self.speed = 0.05  # Speed of the linear trajectory (2 cm/s)

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

        # Timer to periodically publish the trajectory (1 kHz)
        self.timer = self.create_timer(2e-3, self.publish_trajectory)  # 1 ms = 1 kHz

    def publish_trajectory(self):
        # Get the current time
        current_time = time.time()

        # Calculate the elapsed time since the start
        elapsed_time = current_time - self.start_time

        if elapsed_time > self.total_duration:
            self.get_logger().info("Trajectory completed.")
            self.plot_trajectory()
            rclpy.shutdown()
            return

        # Get the current position and orientation of the robot's end effector in the base frame
        try:
            # Get the transform between the end effector and the base link
            transform: TransformStamped = self.tf_buffer.lookup_transform(self.base, self.end_effector, rclpy.time.Time())

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
            # self.get_logger().info(f"Current Pose: {self.current_pose}")
        except Exception as e:
            self.get_logger().warn(f"Could not get the current transform: {e}")
            return

        # Linear trajectory: y(t) = A * t
        commanded_x = self.initial_position[0]  + self.speed * elapsed_time
        commanded_y = self.initial_position[1]  + self.speed * elapsed_time
        commanded_z = self.initial_position[2] 
        self.get_logger().info(f"Commanded Pose: ({commanded_x}, {commanded_y}, {commanded_z})")
        print(f"Delta: {self.speed * elapsed_time}")
        # Create PoseStamped message for commanded trajectory
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = base
        # Set commanded position
        pose.pose.position.x = commanded_x
        pose.pose.position.y = commanded_y
        pose.pose.position.z = commanded_z

        # Set orientation to the original orientation of the end effector
        pose.pose.orientation = self.initial_orientation

        # Publish the message
        self.publisher_.publish(pose)

        # Store the commanded and executed trajectories for later plotting
        self.commanded_trajectory_x.append(commanded_x)
        self.commanded_trajectory_y.append(commanded_y)
        self.commanded_trajectory_z.append(commanded_z)

        self.executed_trajectory_x.append(self.current_pose[0])
        self.executed_trajectory_y.append(self.current_pose[1])
        self.executed_trajectory_z.append(self.current_pose[2])

    def plot_trajectory(self):
        # Plot the commanded and executed trajectories for all axes
        time_steps = np.linspace(0, self.total_duration, len(self.commanded_trajectory_x))

        plt.figure(figsize=(12, 8))

        # Plot x-axis trajectories
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, self.commanded_trajectory_x, label="Commanded X", linestyle='-', color='b')
        plt.plot(time_steps, self.executed_trajectory_x, label="Executed X", linestyle='-', color='r')
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.title("X-Axis: Commanded vs Executed")
        plt.legend()
        plt.grid(True)

        # Plot y-axis trajectories
        plt.subplot(3, 1, 2)
        plt.plot(time_steps, self.commanded_trajectory_y, label="Commanded Y", linestyle='-', color='b')
        plt.plot(time_steps, self.executed_trajectory_y, label="Executed Y", linestyle='-', color='r')
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.title("Y-Axis: Commanded vs Executed")
        plt.legend()
        plt.grid(True)

        # Plot z-axis trajectories
        plt.subplot(3, 1, 3)
        plt.plot(time_steps, self.commanded_trajectory_z, label="Commanded Z", linestyle='-', color='b')
        plt.plot(time_steps, self.executed_trajectory_z, label="Executed Z", linestyle='-', color='r')
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
        error_x = np.array(self.executed_trajectory_x) - np.array(self.commanded_trajectory_x)
        plt.plot(time_steps, error_x, label="Error X", linestyle='-', color='g')
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.title("X-Axis Error")
        plt.legend()
        plt.grid(True)

        # Plot y-axis error
        plt.subplot(3, 1, 2)
        error_y = np.array(self.executed_trajectory_y) - np.array(self.commanded_trajectory_y)
        plt.plot(time_steps, error_y, label="Error Y", linestyle='-', color='g')
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.title("Y-Axis Error")
        plt.legend()
        plt.grid(True)

        # Plot z-axis error
        plt.subplot(3, 1, 3)
        error_z = np.array(self.executed_trajectory_z) - np.array(self.commanded_trajectory_z)
        plt.plot(time_steps, error_z, label="Error Z", linestyle='-', color='g')
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

        self.get_logger().info(f"Mean Square Error - X: {mse_x}, Y: {mse_y}, Z: {mse_z}")
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = LinTrajectoryPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
