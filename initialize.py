ROBOT_TYPE = "kuka"  # "franka" or "kuka" or "z1"


def get_robot_params():
    topic_name = "/target_frame"
    if ROBOT_TYPE == "franka":
        base = "base"
        end_effector = "fr3_hand_tcp"
        prefix = "/cartesian_impedance_controller"
    elif ROBOT_TYPE == "kuka":
        base = "lbr_link_0"
        end_effector = "lbr_link_ee"
        prefix = "lbr"
    elif ROBOT_TYPE == "z1":
        base = "link00"
        end_effector = "link06"
        prefix = "/joint_impedance_controller"
    else:
        print("Robot type unknown")
        exit(1)
    return prefix + topic_name, base, end_effector
