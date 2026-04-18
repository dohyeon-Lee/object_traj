# Usage (run from project root, activate object_traj venv first):
#   source .venv/bin/activate
#   python src/simulate/main.py --no-wandb --pos-only
#   python src/simulate/main.py data/006_mustard_bottle_20200709_143211
#   python src/simulate/main.py data/006_mustard_bottle_20200709_143211 --steps 20 --scale 2.0
#   python src/simulate/main.py data/006_mustard_bottle_20200709_143211 --no-wandb
#   python src/simulate/main.py data/006_mustard_bottle_20200709_143211 --pos-only
#   python src/simulate/main.py data/006_mustard_bottle_20200709_143211 --video-dir videos --project my-project --name my-run

import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np
import wandb
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from scipy.spatial.transform import Rotation

from viz_overlay import get_cam_matrices, draw_axes

CAMERAS     = ["frontview", "birdview", "sideview"]
DATASET_CAM = "dataset_cam"
CAM_H, CAM_W = 480, 640

# Camera-to-robot rotation matrix (empirically verified).
# cam +Z (forward) → robot -X
# cam +Y (down)    → robot -Z
# cam +X (right)   → robot +Y
CAM_TO_ROBOT_R = np.array([[ 0,  0, -1],
                             [ 1,  0,  0],
                             [ 0, -1,  0]], dtype=float)


# ── data ──────────────────────────────────────────────────────────────────────

def load_traj(data_dir):
    """poses.npz → (N,3) pos, (N,4) quat  in camera frame"""
    poses = np.load(Path(data_dir) / "object_pose" / "poses.npz")["poses"]
    pos  = poses[:, :3, 3]
    quat = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    return pos, quat


def cam_to_robot(pos, quat, R=CAM_TO_ROBOT_R):
    """camera frame → robot frame using R (default: CAM_TO_ROBOT_R)"""
    rot = Rotation.from_matrix(R)
    return (R @ pos.T).T, (rot * Rotation.from_quat(quat)).as_quat()


def remap(pos, quat, center=(0.0, 0.0, 1.0), scale=1.0):
    """shift + scale so trajectory center lands at `center` in robosuite world"""
    pos = (pos - pos.mean(axis=0)) * scale + np.array(center)
    return pos, quat


# ── dataset camera pose ───────────────────────────────────────────────────────

def compute_dataset_cam(pos_cam_raw, scale, center=(0.0, 0.0, 1.0), R=CAM_TO_ROBOT_R):
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

    R_cam = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]], dtype=float)
    qxyzw = Rotation.from_matrix(R_cam).as_quat()
    cam_quat_wxyz = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])
    return cam_pos, cam_quat_wxyz


# ── simulation ────────────────────────────────────────────────────────────────

def run_sim(pos, quat, pos_cam_raw, scale, center,
            steps_per_waypoint, video_dir, run_name, wandb_run, pos_only=False):

    cam_pos, cam_quat = compute_dataset_cam(pos_cam_raw, scale, center)

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

    cam_mat    = {cam: get_cam_matrices(env, cam, CAM_H, CAM_W) for cam in CAMERAS}
    frames     = {cam: [] for cam in CAMERAS + [DATASET_CAM]}
    fixed_quat = obs["robot0_eef_quat"].copy()  # initial orientation, used when pos_only

    for i, (tgt_pos, tgt_quat) in enumerate(zip(pos, quat)):
        for _ in range(steps_per_waypoint):

            # ── OSC control: proportional delta toward target ──────────────
            delta_pos = np.clip((tgt_pos - obs["robot0_eef_pos"]) * 5.0, -1, 1)
            rot_target = Rotation.from_quat(fixed_quat if pos_only else tgt_quat)
            r_delta    = rot_target * Rotation.from_quat(obs["robot0_eef_quat"]).inv()
            delta_rot  = np.clip(r_delta.as_rotvec() * 2.0, -1, 1)
            action     = np.concatenate([delta_pos, delta_rot, [-1.0]])  # gripper open
            # ──────────────────────────────────────────────────────────────

            obs, *_ = env.step(action)

            for cam in CAMERAS:
                img = obs[f"{cam}_image"][::-1].copy()
                K, cp, cr = cam_mat[cam]
                draw_axes(img, K, cp, cr, origin=(0, 0, 1), length=0.1, name="(0,0,1)")
                if cam == "birdview":
                    draw_axes(img, K, cp, cr, origin=(0, 0, 0), length=0.1, name="(0,0,0)")
                frames[cam].append(img)

            # Render dataset camera manually (not in robosuite obs pipeline)
            ds_img = env.sim.render(CAM_W, CAM_H, camera_name=DATASET_CAM)
            frames[DATASET_CAM].append(ds_img[::-1])

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
    parser.add_argument("--steps",     type=int,   default=10)
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--project",   default="robosuite-eef-traj")
    parser.add_argument("--name",      default=None)
    parser.add_argument("--no-wandb",  action="store_true")
    parser.add_argument("--pos-only",  action="store_true", help="follow position only, keep initial orientation")
    args = parser.parse_args()

    center    = (0.0, 0.0, 1.0)
    run_name  = args.name or Path(args.data_dir).name
    pos_cam, quat_cam = load_traj(args.data_dir)
    pos, quat = cam_to_robot(pos_cam, quat_cam)
    pos, quat = remap(pos, quat, center=center, scale=args.scale)

    wandb_run = None if args.no_wandb else wandb.init(project=args.project, name=run_name)
    run_sim(pos, quat, pos_cam, args.scale, center,
            args.steps, args.video_dir, run_name, wandb_run, pos_only=args.pos_only)
    if wandb_run:
        wandb_run.finish()
