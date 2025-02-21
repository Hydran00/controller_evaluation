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
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath import UnitQuaternion
from spatialmath.base import tr2eul
from scipy.spatial.transform import Rotation as R


class CTrajPublisher(Node):
    def __init__(self):
        super().__init__("linear_trajectory_publisher")
        self.topic_name, self.base, self.end_effector = get_robot_params()
        # Initialize the publisher for PoseStamped messages
        self.publisher_ = self.create_publisher(PoseStamped, self.topic_name, 10)

        # Initialize tf2 for transforming coordinates
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_freq = 500  # 1 kHz
        self.settling_time = 4.0  # initial and final buffer time

        # Position increment
        self.step_size = np.array([0.15, 0.0, -0.15])
        # Orientation increment in RPY
        self.step_orientation = np.array([-0.0, -0.1, 0.2])
        self.velocity = 0.05  # Speed of the linear trajectory
        
        self.output_img_name = (
            "outputs/cart_traj_-" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(self.step_size) + str(self.step_orientation) + ".png"
        )
        self.output_txt_name = (
            "outputs/cart_traj_-" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(self.step_size) + str(self.step_orientation) + ".txt"
        )

        # Lists to store the commanded and executed trajectories for all axes
        self.commanded_trajectory_x = []
        self.commanded_trajectory_y = []
        self.commanded_trajectory_z = []
        self.executed_trajectory_x = []
        self.executed_trajectory_y = []
        self.executed_trajectory_z = []
        self.commanded_trajectory_R = []
        self.commanded_trajectory_P = []
        self.commanded_trajectory_Y = []
        self.executed_trajectory_R = []
        self.executed_trajectory_P = []
        self.executed_trajectory_Y = []

        # Variable to store current robot pose and orientation
        self.current_pose = None
        self.initial_orientation = None
        self.initial_position = None
        self.cycle_iteration = 0
        self.move_target = False
        self.last_cycle_check = 0
        self.traj_computed = False
        self.traj_idx = 0

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
        self.log_time = [time.time()]

    def get_current_transform(self):
        try:
            transform_msg: TransformStamped = self.tf_buffer.lookup_transform(
                self.base, self.end_effector, rclpy.time.Time()
            )
            return transform_msg
        except Exception as e:
            self.get_logger().warn(f"Failed to get transform: {e}")
            return None

    def compute_ctraj(self):
        # Create a SE3 object from the quaternion and translation vector
        T_start = SE3.Rt(
            UnitQuaternion(
                s=self.initial_orientation[3], v=self.initial_orientation[:3], norm=True
            ).R,
            self.initial_position,
        )
        T_end = SE3.Rt(
            UnitQuaternion(
                s=self.target_orientation[3], v=self.target_orientation[:3], norm=True
            ).R,
            self.target_position,
        )

        distance_between_waypoints = self.velocity / self.pub_freq
        number_of_samples = int(
            np.linalg.norm(self.step_size) / distance_between_waypoints
        )
        print(f"Number of samples: {number_of_samples}")
        # Interpolate between the two SE3 objects
        tg = rtb.ctraj(T_start, T_end, t=number_of_samples)
        print("Computed trajectory, len is ", len(tg))
        # for i in range(0, len(tg)):
        #     print(tg[i].t, " # ", tr2eul(tg[i].R))
        self.trajectory = tg
        self.traj_computed = True

    def publish_trajectory(self):
        self.cycle_iteration += 1

        transform_msg = self.get_current_transform()
        if transform_msg is None:
            return

        # print(f"Cycle iteration: {self.cycle_iteration}")
        if self.state == "initial_waiting":
            # Set the initial orientation and position
            if self.initial_orientation is None:
                self.get_logger().info("Initial waiting...")
                self.initial_orientation = np.array(
                    [
                        transform_msg.transform.rotation.x,
                        transform_msg.transform.rotation.y,
                        transform_msg.transform.rotation.z,
                        transform_msg.transform.rotation.w,
                    ]
                )
                self.initial_position = np.array(
                    [
                        transform_msg.transform.translation.x,
                        transform_msg.transform.translation.y,
                        transform_msg.transform.translation.z,
                    ]
                )
                self.commanded_x = self.initial_position[0]
                self.commanded_y = self.initial_position[1]
                self.commanded_z = self.initial_position[2]
                self.commanded_qx = self.initial_orientation[0]
                self.commanded_qy = self.initial_orientation[1]
                self.commanded_qz = self.initial_orientation[2]
                self.commanded_qw = self.initial_orientation[3]

                self.target_position = self.initial_position + self.step_size
                target_orientation_RPY = self.step_orientation + R.from_quat(
                    self.initial_orientation
                ).as_euler("xyz")
                self.target_orientation = R.from_euler(
                    "xyz", target_orientation_RPY
                ).as_quat()
                # print("Initial orientation: ", self.initial_orientation)
                # print(f"Target orientation: {self.target_orientation}")
                # exit(0)

            if self.cycle_iteration == int(self.settling_time * self.pub_freq):
                self.start_time = time.time()
                self.state = "moving"
                self.get_logger().info("Moving...")
                self.move_target = True

        if self.state == "moving":
            # Get the current time
            current_time = time.time()

            # Calculate the elapsed time since the start
            elapsed_time = current_time - self.start_time
            # if elapsed_time > self.total_duration:
            if self.traj_computed and self.traj_idx >= len(self.trajectory):
                self.move_target = False
                self.state = "final_waiting"
                self.get_logger().info("Final waiting...")

        if self.state == "final_waiting":
            if self.last_cycle_check == int(self.settling_time * self.pub_freq):
                # exit(0)
                # self.get_logger().info("Trajectory completed")
                plot_trajectory(self)
                rclpy.shutdown()
            self.last_cycle_check += 1

        self.current_pose = (
            transform_msg.transform.translation.x,
            transform_msg.transform.translation.y,
            transform_msg.transform.translation.z,
            transform_msg.transform.rotation.x,
            transform_msg.transform.rotation.y,
            transform_msg.transform.rotation.z,
            transform_msg.transform.rotation.w,
        )

        # self.get_logger().info(f"Current Pose: {self.current_pose}")

        if self.move_target:
            if not self.traj_computed:
                self.compute_ctraj()

            traj_point = self.trajectory[self.traj_idx]
            self.traj_idx += 1

            self.commanded_x = traj_point.t[0]
            self.commanded_y = traj_point.t[1]
            self.commanded_z = traj_point.t[2]

            orient = R.from_matrix(traj_point.R).as_quat()
            self.commanded_qx = orient[0]
            self.commanded_qy = orient[1]
            self.commanded_qz = orient[2]
            self.commanded_qw = orient[3]

        # self.get_logger().info(
        #     f"Commanded Pose: ({self.commanded_x}, {self.commanded_y}, {self.commanded_z}), \n orientation: {self.commanded_qx}, {self.commanded_qy}, {self.commanded_qz}, {self.commanded_qw}"
        # )
        # Create PoseStamped message for commanded trajectory
        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = self.base
        # Set commanded position
        target_pose.pose.position.x = self.commanded_x
        target_pose.pose.position.y = self.commanded_y
        target_pose.pose.position.z = self.commanded_z

        # Set orientation to the original orientation of the end effector
        target_pose.pose.orientation.x = self.commanded_qx
        target_pose.pose.orientation.y = self.commanded_qy
        target_pose.pose.orientation.z = self.commanded_qz
        target_pose.pose.orientation.w = self.commanded_qw

        # Publish the message
        self.publisher_.publish(target_pose)

        # Store the commanded and executed trajectories for later plotting
        self.commanded_trajectory_x.append(self.commanded_x)
        self.commanded_trajectory_y.append(self.commanded_y)
        self.commanded_trajectory_z.append(self.commanded_z)
        (self.commanded_R, self.commanded_P, self.commanded_Y) = R.from_quat(
            [self.commanded_qx, self.commanded_qy, self.commanded_qz, self.commanded_qw]
        ).as_euler("xyz")
        self.commanded_trajectory_R.append(self.commanded_R)
        self.commanded_trajectory_P.append(self.commanded_P)
        self.commanded_trajectory_Y.append(self.commanded_Y)

        self.executed_trajectory_x.append(self.current_pose[0])
        self.executed_trajectory_y.append(self.current_pose[1])
        self.executed_trajectory_z.append(self.current_pose[2])
        (self.execute_R, self.execute_P, self.execute_Y) = R.from_quat(
            [
                self.current_pose[3],
                self.current_pose[4],
                self.current_pose[5],
                self.current_pose[6],
            ]
        ).as_euler("xyz")
        self.executed_trajectory_R.append(self.execute_R)
        self.executed_trajectory_P.append(self.execute_P)
        self.executed_trajectory_Y.append(self.execute_Y)
        self.log_time.append(time.time())


def main(args=None):
    rclpy.init(args=args)
    node = CTrajPublisher()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
