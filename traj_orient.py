from initialize import get_robot_params

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import time
import numpy as np
from plot import plot_trajectory
from plot_cone_constraints import plot_cone_constraints
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
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
        self.total_duration = 5.0  # Total duration for trajectory
        self.target_orientation = R.from_euler(
            "xyz", [150.0, 20.0, 100.0], degrees=True
        ).as_quat()

        # Lists to store the commanded and executed trajectories for all axes
        self.commanded_trajectory_x = []
        self.commanded_trajectory_y = []
        self.commanded_trajectory_z = []
        self.executed_trajectory_x = []
        self.executed_trajectory_y = []
        self.executed_trajectory_z = []

        self.log_x = []
        self.log_time = []
        self.log_u = []

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

        current_rot = [
            transform_msg.transform.rotation.x,
            transform_msg.transform.rotation.y,
            transform_msg.transform.rotation.z,
            transform_msg.transform.rotation.w,
        ]
        self.log_x.append(R.from_quat(current_rot).as_matrix())

        if self.state == "initial_waiting":
            # self.get_logger().info("Initial waiting...")
            # Set the initial orientation and position
            if self.initial_orientation is None:
                self.initial_orientation = [
                    transform_msg.transform.rotation.x,
                    transform_msg.transform.rotation.y,
                    transform_msg.transform.rotation.z,
                    transform_msg.transform.rotation.w,
                ]
                self.log_x_init = R.from_quat(self.initial_orientation).as_matrix()
                self.initial_position = [
                    transform_msg.transform.translation.x,
                    transform_msg.transform.translation.y,
                    transform_msg.transform.translation.z,
                ]
                self.commanded_qx = self.initial_orientation[0]
                self.commanded_qy = self.initial_orientation[1]
                self.commanded_qz = self.initial_orientation[2]
                self.commanded_qw = self.initial_orientation[3]

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
                # plot_trajectory(self)
                self.log_time = np.array(self.log_time)
                self.log_x = np.array(self.log_x)
                self.log_x_init = np.array(self.log_x_init)
                self.log_u = np.array(self.log_u)
                plot_cone_constraints(
                    self.log_time,
                    self.log_x,
                    self.log_x_init,
                    None,
                    self.log_u,
                    [0.4, 0.4, 0.4],
                )
                rclpy.shutdown()
            self.last_cycle_check += 1

        # self.current_orientation = (
        #     transform_msg.transform.rotation.x,
        #     transform_msg.transform.rotation.y,
        #     transform_msg.transform.rotation.z,
        #     transform_msg.transform.rotation.w,
        # )

        # self.get_logger().info(f"Current Pose: {self.current_pose}")
        if self.move_target:
            self.log_time.append(elapsed_time)
            # slerp between current and target orientation
            rotations = R.from_quat([self.initial_orientation, self.target_orientation])
            print(rotations)
            slerp = Slerp([0, 1], rotations)
            # get the orientation at the current time
            timestep = 1 / (self.total_duration) * elapsed_time
            if timestep >= 1.0:
                timestep = 0.999999
            print("Timestep:", timestep)
            interpolated_rot = slerp([timestep])
            self.log_u.append(interpolated_rot.as_matrix())
            interpolated_quat = interpolated_rot.as_quat()
            print("Timestep:", timestep, "Interpolated Rot:", interpolated_rot)
            self.commanded_qx = interpolated_quat[0][0]
            self.commanded_qy = interpolated_quat[0][1]
            self.commanded_qz = interpolated_quat[0][2]
            self.commanded_qw = interpolated_quat[0][3]

        # self.get_logger().info(
        #     f"Commanded Pose: ({self.commanded_qx}, {self.commanded_qy}, {self.commanded_qz}, {self.commanded_qw})"
        # )
        # Create PoseStamped message for commanded trajectory
        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = self.base
        # Set commanded position
        target_pose.pose.orientation.x = self.commanded_qx
        target_pose.pose.orientation.y = self.commanded_qy
        target_pose.pose.orientation.z = self.commanded_qz
        target_pose.pose.orientation.w = self.commanded_qw

        # Set orientation to the original orientation of the end effector
        target_pose.pose.position.x = self.initial_position[0]
        target_pose.pose.position.y = self.initial_position[1]
        target_pose.pose.position.z = self.initial_position[2]

        # Publish the message
        self.publisher_.publish(target_pose)

        # Store the commanded and executed trajectories for later plotting

        # convert commanded and current to euler angles for plotting
        eul = R.from_quat(
            [self.commanded_qx, self.commanded_qy, self.commanded_qz, self.commanded_qw]
        ).as_euler("xyz", degrees=True)
        self.commanded_trajectory_x.append(eul[0])
        self.commanded_trajectory_y.append(eul[1])
        self.commanded_trajectory_z.append(eul[2])

        eul = R.from_quat(
            [
                transform_msg.transform.rotation.x,
                transform_msg.transform.rotation.y,
                transform_msg.transform.rotation.z,
                transform_msg.transform.rotation.w,
            ]
        ).as_euler("xyz", degrees=True)
        print("Executed:", eul)
        self.executed_trajectory_x.append(eul[0])
        self.executed_trajectory_y.append(eul[1])
        self.executed_trajectory_z.append(eul[2])


def main(args=None):
    rclpy.init(args=args)
    node = LinTrajectoryPublisher()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
