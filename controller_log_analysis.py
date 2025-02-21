import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_mat_file(filename):
    data = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            dataset = f[key]
            if isinstance(dataset, h5py.Dataset):
                print("Extracting", key, " data: ", f[key].shape)
                data[key] = np.array(dataset, dtype='float64')
            else:
                print(f"Skipping {key}, unexpected data format.")
    return data

def convert_orientation_to_euler(matrix_data):
    euler_angles = np.array([R.from_matrix(m.reshape(3, 3)).as_euler('xyz', degrees=False) for m in matrix_data])
    return euler_angles.T  # Transpose to separate roll, pitch, yaw

def plot_data(data):
    time = data['time_sec'].flatten() + data['time_nsec'].flatten() * 1e-9
    euler_target = convert_orientation_to_euler(data['cart_target_M'])
    euler_current = convert_orientation_to_euler(data['cart_current_M'])
    
    # Plot Cartesian XYZ and RPY angles in one window (two columns)
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    axs[0, 0].plot(time, data['cart_target_x'], 'r--', label="Target X")
    axs[0, 0].plot(time, data['cart_current_x'], 'r', label="Current X")
    axs[0, 0].set_title("Cartesian X")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].legend()

    axs[1, 0].plot(time, data['cart_target_y'], 'g--', label="Target Y")
    axs[1, 0].plot(time, data['cart_current_y'], 'g', label="Current Y")
    axs[1, 0].set_title("Cartesian Y")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].legend()

    axs[2, 0].plot(time, data['cart_target_z'], 'b--', label="Target Z")
    axs[2, 0].plot(time, data['cart_current_z'], 'b', label="Current Z")
    axs[2, 0].set_title("Cartesian Z")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].legend()

    axs[0, 1].plot(time, euler_target[0], 'r--', label="Target Roll")
    axs[0, 1].plot(time, euler_current[0], 'r', label="Current Roll")
    axs[0, 1].set_title("Roll")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].legend()

    axs[1, 1].plot(time, euler_target[1], 'g--', label="Target Pitch")
    axs[1, 1].plot(time, euler_current[1], 'g', label="Current Pitch")
    axs[1, 1].set_title("Pitch")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].legend()

    axs[2, 1].plot(time, euler_target[2], 'b--', label="Target Yaw")
    axs[2, 1].plot(time, euler_current[2], 'b', label="Current Yaw")
    axs[2, 1].set_title("Yaw")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].legend()
    
    fig.tight_layout()  # Adjust subplots to avoid overlap

    # Plot joint position and current in another window (two columns)
    fig2, axs2 = plt.subplots(4, 2, figsize=(12, 16))
    for i in range(7):
        row, col = divmod(i, 2)
        axs2[row, col].plot(time, data['joint_target'][:, i], 'b--', label=f"Joint Target {i+1}")
        axs2[row, col].plot(time, data['joint_current'][:, i], 'b', label=f"Joint Current {i+1}")
        axs2[row, col].set_title(f"Joint {i+1} Position")
        axs2[row, col].set_xlabel("Time (s)")
        axs2[row, col].legend()

    fig2.tight_layout()

    # Plot commanded and measured torques for each joint in another window (two columns)
    fig3, axs3 = plt.subplots(4, 2, figsize=(12, 16))
    for i in range(7):
        row, col = divmod(i, 2)
        axs3[row, col].plot(time, data['commanded_torque'][:, i], 'y--', label=f"Commanded Torque {i+1}")
        axs3[row, col].plot(time, data['measured_torque'][:, i], 'y', label=f"Measured Torque {i+1}")
        axs3[row, col].set_title(f"Joint {i+1} Torques")
        axs3[row, col].set_xlabel("Time (s)")
        axs3[row, col].legend()

    fig3.tight_layout()

    # Plot Jacobian condition number over time  
    fig4, ax4 = plt.subplots()
    ax4.plot(time, data['condition_number_jac'], 'm', label="Jacobian Condition Number")
    ax4.set_title("Jacobian Condition Number")
    ax4.set_xlabel("Time (s)")
    ax4.legend()


    # Show all figures
    plt.show()


if __name__ == "__main__":
    filename = "/tmp/cart_impedance.mat"  # Change this to your actual file name
    data = load_mat_file(filename)
    plot_data(data)

