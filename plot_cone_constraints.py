from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_cone_constraints(time, x_ref, x, h, u, theta):
    # Create a figure with multiple subplots (2 rows, 2 columns)
    fig = plt.figure(figsize=(20, 20))

    # Plot the barrier function values over time (subplot 1)
    # if h is not None:
    #     ax1 = fig.add_subplot(2, 2, 1)
    #     ax1.plot(
    #         time,
    #         h[0, :],
    #         label=f"Barrier function (axis X)",
    #         linewidth=2.5,
    #         color="red",
    #     )
    #     ax1.plot(
    #         time,
    #         h[1, :],
    #         label=f"Barrier function (axis Y)",
    #         linewidth=2.5,
    #         color="green",
    #     )
    #     ax1.plot(
    #         time,
    #         h[2, :],
    #         label=f"Barrier function (axis Z)",
    #         linewidth=2.5,
    #         color="blue",
    #     )

    #     ax1.axhline(0, color="black", linestyle="--", label="h = 0", linewidth=2.5)
    #     ax1.set_xlabel("Time (s)", fontsize=20)
    #     ax1.set_ylabel("h(R)", fontsize=20)
    #     ax1.set_title("CBF Values Over Time", fontsize=24)
    #     ax1.legend(fontsize=16)
    #     ax1.grid(True)

    # # Plot the control inputs over time (subplot 2)
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax2.plot(
    #     time, u[0, :], label=f"Control input (omega X)", linewidth=2.5, color="red"
    # )
    # ax2.plot(
    #     time, u[1, :], label=f"Control input (omega Y)", linewidth=2.5, color="green"
    # )
    # ax2.plot(
    #     time, u[2, :], label=f"Control input (omega X)", linewidth=2.5, color="blue"
    # )

    # ax2.set_xlabel("Time (s)", fontsize=20)
    # ax2.set_ylabel("Control input (rad/s)", fontsize=20)
    # ax2.set_title("Control Inputs Over Time", fontsize=24)
    # ax2.legend(fontsize=16)
    # ax2.grid(True)

    # ax3 = fig.add_subplot(1, 1, 1, projection="3d")

    # Function to plot a half-cone along a specified axis
    def plot_half_cone(ax, theta, direction, color, alpha=0.3):
        cone_height = 1.0  # Height of the cone (unit length for visualization)
        u, v = np.mgrid[0 : 2 * np.pi : 100j, 0:cone_height:50j]

        # Create half-cone surface (opening angle based on theta)
        radius = v * np.tan(theta)
        x = radius * np.cos(u)
        y = radius * np.sin(u)
        z = v

        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # Find a rotation matrix that aligns the z-axis with the given direction
        z_axis = np.array([0, 0, 1])  # Default axis of the cone
        v = np.cross(z_axis, direction)  # Find the axis of rotation
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)

        # If the direction is the same as z-axis, no rotation is needed
        if s == 0:
            if c < 0:
                R = -np.eye(3)
            else:
                R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / s**2)

        # Apply the rotation matrix to the cone points
        cone_points = np.vstack((x.ravel(), y.ravel(), z.ravel()))
        rotated_points = R @ cone_points
        x_rot, y_rot, z_rot = rotated_points.reshape(3, *x.shape)

        # Plot the rotated cone
        ax.plot_surface(x_rot, y_rot, z_rot, color=color, alpha=alpha)

    # Plot the 3D dynamics (subplot 3)
    ax3 = fig.add_subplot(1, 1, 1, projection="3d")

    # Plot initial frame
    ax3.quiver(
        0,
        0,
        0,
        x[0, 0, 0],
        x[0, 1, 0],
        x[0, 2, 0],
        color="r",
        linewidth=2.5,
        label="Initial frame (x-axis)",
    )
    ax3.quiver(
        0,
        0,
        0,
        x[0, 0, 1],
        x[0, 1, 1],
        x[0, 2, 1],
        color="g",
        linewidth=2.5,
        label="Initial frame (y-axis)",
    )
    ax3.quiver(
        0,
        0,
        0,
        x[0, 0, 2],
        x[0, 1, 2],
        x[0, 2, 2],
        color="b",
        linewidth=2.5,
        label="Initial frame (z-axis)",
    )

    # Plot final frame
    ax3.quiver(
        0,
        0,
        0,
        x[-1, 0, 0],
        x[-1, 1, 0],
        x[-1, 2, 0],
        color="r",
        linestyle="dashed",
        linewidth=2.5,
        label="Final frame (x-axis)",
    )
    ax3.quiver(
        0,
        0,
        0,
        x[-1, 0, 1],
        x[-1, 1, 1],
        x[-1, 2, 1],
        color="g",
        linestyle="dashed",
        linewidth=2.5,
        label="Final frame (y-axis)",
    )
    ax3.quiver(
        0,
        0,
        0,
        x[-1, 0, 2],
        x[-1, 1, 2],
        x[-1, 2, 2],
        color="b",
        linestyle="dashed",
        linewidth=2.5,
        label="Final frame (z-axis)",
    )

    # Plot the trajectory of x on the unit sphere (projection of x)
    for t in range(0, x.shape[0], 100):
        norm_x = np.linalg.norm(x[t, :, 0])  # Normalize each x to have a radius = 1
        x_proj = x[t, :, 0] / norm_x  # Projection on unit sphere

        norm_y = np.linalg.norm(x[t, :, 1])
        y_proj = x[t, :, 1] / norm_y

        norm_z = np.linalg.norm(x[t, :, 2])
        z_proj = x[t, :, 2] / norm_z

        norm_x_commanded = np.linalg.norm(
            u[t, :, 0]
        )  # Normalize each x to have a radius = 1
        x_proj_commanded = u[t, :, 0] / norm_x_commanded  # Projection on unit sphere

        norm_y_commanded = np.linalg.norm(u[t, :, 1])
        y_proj_commanded = u[t, :, 1] / norm_y_commanded

        norm_z_commanded = np.linalg.norm(u[t, :, 2])
        z_proj_commanded = u[t, :, 2] / norm_z_commanded

        # plot projection points
        ax3.scatter(x_proj[0], x_proj[1], x_proj[2], color="r", s=30)
        ax3.scatter(y_proj[0], y_proj[1], y_proj[2], color="g", s=30)
        ax3.scatter(z_proj[0], z_proj[1], z_proj[2], color="b", s=30)

        ax3.scatter(
            x_proj_commanded[0],
            x_proj_commanded[1],
            x_proj_commanded[2],
            color="darkred",
            marker="x",
            s=30,
        )
        ax3.scatter(
            y_proj_commanded[0],
            y_proj_commanded[1],
            y_proj_commanded[2],
            color="darkgreen",
            marker="x",
            s=30,
        )
        ax3.scatter(
            z_proj_commanded[0],
            z_proj_commanded[1],
            z_proj_commanded[2],
            color="darkblue",
            marker="x",
            s=30,
        )

    # Add half-cones for each axis
    plot_half_cone(ax3, theta[0], x_ref[:, 0], "red", alpha=0.3)
    plot_half_cone(ax3, theta[1], x_ref[:, 1], "green", alpha=0.3)
    plot_half_cone(ax3, theta[2], x_ref[:, 2], "blue", alpha=0.3)

    # Set axis limits and labels
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([-1, 1])
    ax3.set_xlabel("X", fontsize=20)
    ax3.set_ylabel("Y", fontsize=20)
    ax3.set_zlabel("Z", fontsize=20)
    ax3.set_title("3D Visualization of Rotation Dynamics with Cone Limits", fontsize=24)
    ax3.legend(fontsize=16)

    # Adjust layout and show the figure with all plots
    plt.tight_layout()
    plt.show()
