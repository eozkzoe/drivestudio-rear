# Camera pose manipulation and trajectory generation.
import os
import torch
import numpy as np
from typing import Dict

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


def interpolate_poses(key_poses: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Interpolate between key poses to generate a smooth trajectory.

    Args:
        key_poses (torch.Tensor): Tensor of shape (N, 4, 4) containing key camera poses.
        target_frames (int): Number of frames to interpolate.

    Returns:
        torch.Tensor: Interpolated poses of shape (target_frames, 4, 4).
    """
    device = key_poses.device
    key_poses = key_poses.cpu().numpy()

    # Separate translation and rotation
    translations = key_poses[:, :3, 3]
    rotations = key_poses[:, :3, :3]

    # Create time array
    times = np.linspace(0, 1, len(key_poses))
    target_times = np.linspace(0, 1, target_frames)

    # Interpolate translations
    interp_translations = np.stack(
        [np.interp(target_times, times, translations[:, i]) for i in range(3)], axis=-1
    )

    # Interpolate rotations using Slerp
    key_rots = R.from_matrix(rotations)
    slerp = Slerp(times, key_rots)
    interp_rotations = slerp(target_times).as_matrix()

    # Combine interpolated translations and rotations
    interp_poses = np.eye(4)[None].repeat(target_frames, axis=0)
    interp_poses[:, :3, :3] = interp_rotations
    interp_poses[:, :3, 3] = interp_translations

    return torch.tensor(interp_poses, dtype=torch.float32, device=device)


def look_at_rotation(
    direction: torch.Tensor, up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])
) -> torch.Tensor:
    """Calculate rotation matrix to look at a specific direction."""
    front = torch.nn.functional.normalize(direction, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(front, up), dim=-1)
    up = torch.cross(right, front)
    rotation_matrix = torch.stack([right, up, -front], dim=-1)
    return rotation_matrix


def get_interp_novel_trajectories(
    dataset_type: str,
    scene_idx: str,
    per_cam_poses: Dict[int, torch.Tensor],
    traj_type: str = "three_key_poses",
    target_frames: int = 100,
) -> torch.Tensor:
    original_frames = per_cam_poses[list(per_cam_poses.keys())[0]].shape[0]

    trajectory_generators = {
        "front_center_interp": front_center_interp,
        "s_curve": s_curve,
        "three_key_poses": three_key_poses_trajectory,
    }

    if traj_type not in trajectory_generators:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return trajectory_generators[traj_type](
        dataset_type, per_cam_poses, original_frames, target_frames
    )


def front_center_interp(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    num_loops: int = 1,
) -> torch.Tensor:
    # """Interpolate key frames from the front center camera."""
    # assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required for front_center_interp"
    # key_poses = per_cam_poses[0][::original_frames//4]  # Select every 4th frame as key frame
    # return interpolate_poses(key_poses, target_frames)
    # assert 0 in per_cam_poses.keys() or 1 in per_cam_poses.keys(), "At least one of camera 0 or camera 1 is required"

    # # Randomly choose between camera 0 and camera 1
    # chosen_cam = 0 if (0 in per_cam_poses and (1 not in per_cam_poses or np.random.rand() < 0.5)) else 1

    # key_poses = per_cam_poses[chosen_cam][::original_frames // 4]  # Select every 4th frame as key frame
    # return interpolate_poses(key_poses, target_frames)
    assert (
        0 in per_cam_poses.keys() and 1 in per_cam_poses.keys()
    ), "Both cameras 0 and 1 are required"

    key_poses = []
    for i in range(0, original_frames, 4):
        if i % 8 == 0 and 0 in per_cam_poses:
            key_poses.append(per_cam_poses[0][i])
        elif 1 in per_cam_poses:
            key_poses.append(per_cam_poses[1][i])

    key_poses = torch.stack(key_poses, dim=0)

    return interpolate_poses(key_poses, target_frames)


def s_curve(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """Create an S-shaped trajectory using the front three cameras."""
    left_cam, right_cam, back_cam = 0, 0, 0
    if not all(cam in per_cam_poses.keys() for cam in [0, 1, 2]):
        if dataset_type == "nuscenes":
            assert all(cam in per_cam_poses.keys() for cam in [3, 4, 5])
            left_cam = 3
            right_cam = 4
            back_cam = 5

        elif dataset_type == "nuplan":
            assert all(cam in per_cam_poses.keys() for cam in [5, 6, 7])
            left_cam = 5
            right_cam = 6
            back_cam = 7

        else:
            raise AssertionError(
                "Rear cam s_curve only supported for nuscnes and nuplan!"
            )

        assert all(
            (left_cam, right_cam, back_cam)
        ), "Error in loading rear cams for s_curve generation"

        print("Warning: Rear cam is experimental")
        key_poses = torch.cat(
            [
                per_cam_poses[back_cam][0:1],
                per_cam_poses[right_cam][
                    original_frames // 4 : original_frames // 4 + 1
                ],
                per_cam_poses[back_cam][
                    original_frames // 2 : original_frames // 2 + 1
                ],
                per_cam_poses[left_cam][
                    3 * original_frames // 4 : 3 * original_frames // 4 + 1
                ],
                per_cam_poses[back_cam][-1:],
            ],
            dim=0,
        )
    else:
        key_poses = torch.cat(
            [
                per_cam_poses[0][0:1],
                per_cam_poses[1][original_frames // 4 : original_frames // 4 + 1],
                per_cam_poses[0][original_frames // 2 : original_frames // 2 + 1],
                per_cam_poses[2][
                    3 * original_frames // 4 : 3 * original_frames // 4 + 1
                ],
                per_cam_poses[0][-1:],
            ],
            dim=0,
        )
    print("s_curve successfully initialised")
    return interpolate_poses(key_poses, target_frames)


def three_key_poses_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """
    Create a trajectory using three key poses:
    1. First frame of front center camera
    2. Middle frame with interpolated rotation and position from camera 1 or 2
    3. Last frame of front center camera

    The rotation of the middle pose is calculated using Slerp between
    the start frame and the middle frame of camera 1 or 2.

    Args:
        dataset_type (str): Type of the dataset (e.g., "waymo", "pandaset", etc.).
        per_cam_poses (Dict[int, torch.Tensor]): Dictionary of camera poses.
        original_frames (int): Number of original frames.
        target_frames (int): Number of frames in the output trajectory.

    Returns:
        torch.Tensor: Trajectory of shape (target_frames, 4, 4).
    """
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required"
    assert (
        1 in per_cam_poses.keys() or 2 in per_cam_poses.keys()
    ), "Either camera 1 or camera 2 is required"

    # First key pose: First frame of front center camera
    start_pose = per_cam_poses[0][0]
    key_poses = [start_pose]

    # Select camera for middle frame
    middle_frame = int(original_frames // 2)
    chosen_cam = np.random.choice([1, 2])

    middle_pose = per_cam_poses[chosen_cam][middle_frame]

    # Calculate interpolated rotation for middle pose
    start_rotation = R.from_matrix(start_pose[:3, :3].cpu().numpy())
    middle_rotation = R.from_matrix(middle_pose[:3, :3].cpu().numpy())
    slerp = Slerp(
        [0, 1], R.from_quat([start_rotation.as_quat(), middle_rotation.as_quat()])
    )
    interpolated_rotation = slerp(0.5).as_matrix()

    # Create middle key pose with interpolated rotation and original translation
    middle_key_pose = torch.eye(4, device=start_pose.device)
    middle_key_pose[:3, :3] = torch.tensor(
        interpolated_rotation, device=start_pose.device
    )
    middle_key_pose[:3, 3] = middle_pose[:3, 3]  # Keep the original translation
    key_poses.append(middle_key_pose)

    # Third key pose: Last frame of front center camera
    key_poses.append(per_cam_poses[0][-1])

    # Stack the key poses and interpolate
    key_poses = torch.stack(key_poses)
    return interpolate_poses(key_poses, target_frames)

