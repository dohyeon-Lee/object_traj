# Usage (run from project root, activate object_traj venv first):
#   source .venv/bin/activate
#   python src/main.py --no-wandb --show-eef
#   python src/main.py data/011_banana_20200709_145401 --no-wandb
#   python src/main.py data/011_banana_20200709_145401 --steps 20 --scale 2.0
#   python src/main.py data/011_banana_20200709_145401 --angle 45 --no-wandb

import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Project root = one level up from src/main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np
import wandb
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from scipy.spatial.transform import Rotation

from viz_overlay import get_cam_matrices, draw_eef, make_video

CAMERAS     = ["frontview", "birdview", "sideview"]
DATASET_CAM = "dataset_cam"
CAM_H, CAM_W = 480, 640

# Camera-to-robot rotation matrix (empirically verified).
# Assumption: dataset camera faces the robot head-on in simulation (+X direction).
# cam +Z (optical axis) = robot -X
# cam +Y (down)         = robot -Z
# cam +X (right)        = robot +Y
DATASET_CAM_FRAME_IN_ROBOT = np.array( [[ 0,  0, -1],
                                        [ 1,  0,  0],
                                        [ 0, -1,  0]], dtype=float)

KP_POS = 5.0   # proportional gain for position error (m → action)
KP_ROT = 2.0   # proportional gain for rotation error (rad → action)


def cam_to_robot_matrix(angle_deg: float, R0=DATASET_CAM_FRAME_IN_ROBOT) -> np.ndarray:
    """Cam-to-robot rotation matrix for a given horizontal camera angle.

    Angle is defined from the robot's perspective:
    0°   = camera directly in front (head-on)
    +90° = camera to the robot's right
    -90° = camera to the robot's left
    """
    rot_z = Rotation.from_euler('z', -angle_deg, degrees=True).as_matrix()
    return rot_z @ R0


# ── data ──────────────────────────────────────────────────────────────────────

def load_traj(data_dir):
    """poses.npz → (N,3) pos, (N,4) quat  in camera frame"""
    poses = np.load(Path(data_dir) / "object_pose" / "poses.npz")["poses"]
    pos  = poses[:, :3, 3]
    quat = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    return pos, quat


def cam_to_robot(pos, quat, R=DATASET_CAM_FRAME_IN_ROBOT):
    """camera frame → robot frame using R (default: DATASET_CAM_FRAME_IN_ROBOT)"""
    rot = Rotation.from_matrix(R)
    return (R @ pos.T).T, (rot * Rotation.from_quat(quat)).as_quat()


def remap(pos, quat, center=(0.0, 0.0, 1.0), scale=1.0):
    """shift + scale so trajectory center lands at `center` in robosuite world"""
    pos = (pos - pos.mean(axis=0)) * scale + np.array(center)
    return pos, quat


# ── dataset camera pose ───────────────────────────────────────────────────────

def compute_dataset_cam(pos_cam_raw, scale, center=(0.0, 0.0, 1.0), R=DATASET_CAM_FRAME_IN_ROBOT):
    """Compute dataset camera position and MuJoCo quaternion in robot frame.

    The camera origin (0,0,0 in cam frame) undergoes the same remap as the
    trajectory points, telling us where the real camera sits in robot world.

    MuJoCo camera convention: looks along -Z_cam, up is +Y_cam.
    Dataset camera: looks along -X_robot, up is +Z_robot.
    Rotation matrix (columns = cam axes in robot frame):
        X_cam = +Y_robot = [0,1,0]
        Y_cam = +Z_robot = [0,0,1]
        Z_cam = +X_robot = [1,0,0]  (camera backward = +X, so forward = -X)
    → quaternion [w,x,y,z] = [0.5, 0.5, 0.5, 0.5]
    """
    mean_raw = (R @ pos_cam_raw.T).T.mean(axis=0)
    cam_pos  = (np.zeros(3) - mean_raw) * scale + np.array(center)

    # MuJoCo cam frame from R (columns = cam axes in robot frame):
    #   X_cam=R[:,0], Y_cam(up)=-R[:,1], Z_cam(backward)=-R[:,2]
    mujoco_cam_mat = np.column_stack([R[:, 0], -R[:, 1], -R[:, 2]])
    qxyzw = Rotation.from_matrix(mujoco_cam_mat).as_quat()
    cam_quat_wxyz = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])
    return cam_pos, cam_quat_wxyz


# ── simulation ────────────────────────────────────────────────────────────────

def run_sim(pos, quat, pos_cam_raw, scale, center,
            steps_per_waypoint, video_dir, run_name, wandb_run, R=DATASET_CAM_FRAME_IN_ROBOT,
            angle=0.0, show_eef=False):

    cam_pos, cam_quat = compute_dataset_cam(pos_cam_raw, scale, center, R=R)
    dataset_cam_key = f"dataset_cam_{angle:g}"

    # hard_reset=False: reset() calls sim.reset() only, not _load_model()/_initialize_sim()
    # This lets us inject a custom camera via _initialize_sim(new_xml) without it being overwritten.
    env = suite.make(
        env_name="Lift", robots="Panda",
        controller_configs=load_composite_controller_config(robot="panda"),
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERAS, camera_heights=CAM_H, camera_widths=CAM_W,
        ignore_done=True, horizon=steps_per_waypoint * len(pos) + 100,
        hard_reset=False,
    )

    # Inject dataset camera: parse current XML, add camera, recompile
    xml  = env.model.get_xml()
    root = ET.fromstring(xml)
    ET.SubElement(root.find('worldbody'), 'camera', {
        'name': DATASET_CAM,
        'pos':  ' '.join(f"{v:.4f}" for v in cam_pos),
        'quat': ' '.join(f"{v:.6f}" for v in cam_quat),
        'fovy': '60',
    })
    env._initialize_sim(ET.tostring(root, encoding='unicode'))

    obs = env.reset()  # hard_reset=False → sim.reset() only, keeps our compiled model
    print(f"Dataset cam pos (robot frame): {cam_pos.round(3)}")

    cam_mat    = {cam: get_cam_matrices(env, cam, CAM_H, CAM_W) for cam in CAMERAS + [DATASET_CAM]}
    frames     = {cam: [] for cam in CAMERAS + [dataset_cam_key]}
    rot_eef_init      = Rotation.from_quat(obs["robot0_eef_quat"].copy())
    rot_dataset_first = Rotation.from_quat(quat[0])
    C_inv = rot_eef_init.inv() * rot_dataset_first  # corrects EEF frame → dataset object frame
    eef_history = []  # world-frame EEF positions for trail

    for i, (tgt_pos, tgt_quat) in enumerate(zip(pos, quat)):
        for _ in range(steps_per_waypoint):

            # ── OSC control: proportional delta toward target ──────────────
            delta_pos  = np.clip((tgt_pos - obs["robot0_eef_pos"]) * KP_POS, -1, 1)
            # world-frame delta from dataset frame 0 → i, applied on top of EEF init
            rot_target = Rotation.from_quat(tgt_quat) * rot_dataset_first.inv() * rot_eef_init
            r_delta    = rot_target * Rotation.from_quat(obs["robot0_eef_quat"]).inv()
            delta_rot  = np.clip(r_delta.as_rotvec() * KP_ROT, -1, 1)
            action     = np.concatenate([delta_pos, delta_rot, [-1.0]])  # gripper open
            # ──────────────────────────────────────────────────────────────

            obs, *_ = env.step(action)
            eef_history.append(obs["robot0_eef_pos"].copy())

            eef_quat_vis = (Rotation.from_quat(obs["robot0_eef_quat"]) * C_inv).as_quat()
            for cam in CAMERAS:
                img = obs[f"{cam}_image"][::-1].copy()
                K, cp, cr = cam_mat[cam]
                if show_eef:
                    draw_eef(img, eef_history, eef_quat_vis, K, cp, cr)
                frames[cam].append(img)

            # Render dataset camera manually (not in robosuite obs pipeline)
            ds_img = env.sim.render(CAM_W, CAM_H, camera_name=DATASET_CAM)[::-1].copy()
            if show_eef:
                K, cp, cr = cam_mat[DATASET_CAM]
                draw_eef(ds_img, eef_history, eef_quat_vis, K, cp, cr)
            frames[dataset_cam_key].append(ds_img)

        print(f"[{i+1}/{len(pos)}] err={np.linalg.norm(obs['robot0_eef_pos'] - tgt_pos):.4f}")

    env.close()
    _save(frames, video_dir, run_name, wandb_run)


def _save(frames, video_dir, run_name, wandb_run):
    out_dir = Path(video_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam, cam_frames in frames.items():
        path = str(out_dir / f"{cam}.mp4")
        imageio.mimwrite(path, cam_frames, fps=20, codec="libx264",
                         output_params=["-crf", "18"])
        print(f"Saved {cam} -> {path}")
        if wandb_run:
            wandb_run.log({f"video/{cam}": wandb.Video(path, fps=20, format="mp4")})


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",    nargs="?", default="data/006_mustard_bottle_20200709_143211")
    parser.add_argument("--scale",     type=float, default=1)
    parser.add_argument("--steps",     type=int,   default=2)
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--project",   default="robosuite-eef-traj")
    parser.add_argument("--name",      default=None)
    parser.add_argument("--no-wandb",  action="store_true")
    parser.add_argument("--show-eef",  action="store_true", help="overlay EEF trail and orientation axes on video")
    parser.add_argument("--angle",     type=float, default=0.0, help="horizontal camera angle in degrees (0=head-on, +90=right, -90=left)")
    args = parser.parse_args()

    center    = (0.0, 0.0, 1.0)
    data_dir  = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    run_name  = args.name or data_dir.name
    R = cam_to_robot_matrix(args.angle)
    pos_cam, quat_cam = load_traj(data_dir)
    pos, quat = cam_to_robot(pos_cam, quat_cam, R=R)
    pos, quat = remap(pos, quat, center=center, scale=args.scale)

    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = PROJECT_ROOT / video_dir

    make_video(data_dir)

    wandb_run = None if args.no_wandb else wandb.init(project=args.project, name=run_name)
    run_sim(pos, quat, pos_cam, args.scale, center,
            args.steps, video_dir, run_name, wandb_run, R=R,
            angle=args.angle, show_eef=args.show_eef)
    if wandb_run:
        wandb_run.finish()
