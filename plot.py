import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(self):
    # Plot the commanded and executed trajectories for all axes
    time_steps = self.log_time[
        0 : len(self.commanded_trajectory_x)
    ]  # np.linspace(0, self.total_duration, len(self.commanded_trajectory_x))

    # Calculate the global y-range across all trajectories
    all_trajectories = np.concatenate(
        [
            self.commanded_trajectory_x,
            self.executed_trajectory_x,
            self.commanded_trajectory_y,
            self.executed_trajectory_y,
            self.commanded_trajectory_z,
            self.executed_trajectory_z,
        ]
    )
    y_min, y_max = np.min(all_trajectories), np.max(all_trajectories)

    # Adjust y-limits symmetrically
    # # check has attribute
    # if hasattr(self, 'y_lim_x'):) = (y_min + y_max) / 2
    # y_range = (y_max - y_min) / 2
    # y_lim = (y_center - y_range * 1.05, y_center + y_range * 1.05)  # Add a 10% margin

    plt.figure(figsize=(12, 8))

    # Plot x-axis trajectories
    plt.subplot(3, 2, 1)
    plt.plot(
        time_steps,
        self.commanded_trajectory_x,
        label="Commanded X",
        linestyle="--",
        color="b",
    )
    plt.plot(
        time_steps,
        self.executed_trajectory_x,
        label="Executed X",
        linestyle="-",
        color="r",
    )
    # check has attribute
    if hasattr(self, "y_lim_x"):
        plt.ylim(self.y_lim_x)
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("X-Axis: Commanded vs Executed")
    plt.legend()
    plt.grid(True)

    # Plot y-axis trajectories
    plt.subplot(3, 2, 3)
    plt.plot(
        time_steps,
        self.commanded_trajectory_y,
        label="Commanded Y",
        linestyle="--",
        color="b",
    )
    plt.plot(
        time_steps,
        self.executed_trajectory_y,
        label="Executed Y",
        linestyle="-",
        color="r",
    )
    # check has attribute
    if hasattr(self, "y_lim_y"):
        plt.ylim(self.y_lim_y)
    plt.ylabel("Position [m]")
    plt.title("Y-Axis: Commanded vs Executed")
    plt.legend()
    plt.grid(True)

    # Plot z-axis trajectories
    plt.subplot(3, 2, 5)
    plt.plot(
        time_steps,
        self.commanded_trajectory_z,
        label="Commanded Z",
        linestyle="--",
        color="b",
    )
    plt.plot(
        time_steps,
        self.executed_trajectory_z,
        label="Executed Z",
        linestyle="-",
        color="r",
    )
    if hasattr(self, "y_lim_z"):
        plt.ylim(self.y_lim_z)
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Z-Axis: Commanded vs Executed")
    plt.legend()
    plt.grid(True)

    if hasattr(self, "commanded_trajectory_R") and hasattr(self, "executed_trajectory_R"):
        plt.subplot(3, 2, 2)
        plt.plot(
            time_steps,
            self.commanded_trajectory_R,
            label="Commanded R",
            linestyle="--",
            color="g",
        )
        plt.plot(
            time_steps,
            self.executed_trajectory_R,
            label="Executed R",
            linestyle="-",
            color="y",
        )
        if hasattr(self, "y_lim_R"):
            plt.ylim(self.y_lim_R)
        plt.ylabel("Position [m]")
        plt.title("Roll: Commanded vs Executed")
        plt.legend()
        plt.grid(True)
    if hasattr(self, "commanded_trajectory_P") and hasattr(self, "executed_trajectory_P"):
        plt.subplot(3, 2, 4)
        plt.plot(
            time_steps,
            self.commanded_trajectory_P,
            label="Commanded P",
            linestyle="--",
            color="g",
        )
        plt.plot(
            time_steps,
            self.executed_trajectory_P,
            label="Executed P",
            linestyle="-",
            color="y",
        )
        if hasattr(self, "y_lim_P"):
            plt.ylim(self.y_lim_P)
        plt.ylabel("Position [m]")
        plt.title("Pitch: Commanded vs Executed")
        plt.legend()
        plt.grid(True)
    if hasattr(self, "commanded_trajectory_Y") and hasattr(self, "executed_trajectory_Y"):
        plt.subplot(3, 2, 6)
        plt.plot(
            time_steps,
            self.commanded_trajectory_Y,
            label="Commanded Y",
            linestyle="--",
            color="g",
        )
        plt.plot(
            time_steps,
            self.executed_trajectory_Y,
            label="Executed Y",
            linestyle="-",
            color="y",
        )
        if hasattr(self, "y_lim_Y"):
            plt.ylim(self.y_lim_Y)
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.title("Yaw: Commanded vs Executed")
        plt.legend()
        plt.grid(True)

    # Show all plots
    plt.tight_layout()
    # Calculate and print the mean square error for each direction
    error_x = np.array(self.executed_trajectory_x) - np.array(
        self.commanded_trajectory_x
    )
    error_y = np.array(self.executed_trajectory_y) - np.array(
        self.commanded_trajectory_y
    )
    error_z = np.array(self.executed_trajectory_z) - np.array(
        self.commanded_trajectory_z
    )
    mse_x = np.mean(np.square(error_x))
    mse_y = np.mean(np.square(error_y))
    mse_z = np.mean(np.square(error_z))
    self.get_logger().info(f"Mean Square Error - X: {mse_x}, Y: {mse_y}, Z: {mse_z}")
    # save error in file
    with open(self.output_txt_name, "w") as f:
        f.write(f"Mean Square Error - X: {mse_x}, Y: {mse_y}, Z: {mse_z}")
    # save the plot
    plt.savefig(self.output_img_name)
    plt.show()
