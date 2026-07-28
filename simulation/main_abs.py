# Usage (run from project root, activate object_traj venv first):
#   source .venv/bin/activate
#   python src/main.py data/freepose --no-wandb --show-eef --angle 45 --eef-dir my
#   python simulation/main.py data/bowl6 --show-eef 
import os
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _load_config():
    path = PROJECT_ROOT / "config.yml"
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}
os.environ.setdefault("MUJOCO_GL", "egl")

import re
import imageio
import numpy as np
import wandb
import robosuite as suite
from robosuite.models.robots import Panda as PandaModel
from robosuite.models.robots.robot_model import register_robot
from robosuite.robots import ROBOT_CLASS_MAPPING
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Panda model + Robotiq85 with 45° z-axis rotation at gripper mount
class PandaRotatedGripper(PandaModel):
    @property
    def gripper_mount_quat_offset(self):
        q = Rotation.from_euler('z', 45, degrees=True).as_quat()  # scipy [x,y,z,w]
        return {'right': np.array([q[3], q[0], q[1], q[2]])}      # MuJoCo [w,x,y,z]

register_robot(PandaRotatedGripper)
ROBOT_CLASS_MAPPING['PandaRotatedGripper'] = ROBOT_CLASS_MAPPING['Panda']

try:
    from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
    def _load_ctrl_cfg():
        cfg = load_composite_controller_config(robot="panda")
        for part_cfg in cfg.get("body_parts", {}).values():
            if isinstance(part_cfg, dict) and part_cfg.get("type", "").upper().startswith("OSC"):
                part_cfg["input_type"] = "absolute"
                part_cfg["input_ref_frame"] = "world"
        return cfg
except Exception:
    from robosuite.controllers import load_controller_config
    def _load_ctrl_cfg():
        cfg = load_controller_config(default_controller="OSC_POSE")
        cfg["input_type"] = "absolute"
        cfg["input_ref_frame"] = "world"
        return cfg

from viz_overlay import (get_cam_matrices, draw_eef,
                        inject_object_xml, make_object_updater, make_video)

# ── constants ─────────────────────────────────────────────────────────────────

CAMERAS     = ["frontview", "birdview", "sideview"]
DATASET_CAM = "dataset_cam"

# cam +Z (optical) = robot -X,  cam +Y (down) = robot -Z,  cam +X (right) = robot +Y
DATASET_CAM_FRAME_IN_ROBOT = np.array([[0, 0, -1],
                                        [1, 0,  0],
                                        [0, -1, 0]], dtype=float)


DROID_INIT_QPOS = np.array([0, -np.pi / 5, 0, -4 * np.pi / 5, 0, 3 * np.pi / 5, 0.0])
HAND_JOINT_Z_OFFSET = Rotation.from_euler('z', -np.pi / 4)

def _set_droid_init_qpos(env):
    import mujoco
    robot = env.robots[0]
    env.sim.data.qpos[robot._ref_joint_pos_indexes] = DROID_INIT_QPOS
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0
    m = getattr(env.sim.model, '_model', env.sim.model)
    d = getattr(env.sim.data, '_data', env.sim.data)
    mujoco.mj_forward(m, d)

_EEF_DIR_ROT = {
    'mz': Rotation.identity(),
    'py': Rotation.from_euler('y',  90, degrees=True) * Rotation.from_euler('x',  90, degrees=True),
    'my': Rotation.from_euler('y',  90, degrees=True) * Rotation.from_euler('x', -90, degrees=True),
}

def _parse_eef_dir(s):
    """Parse --eef-dir: 'mz'/'py'/'my' or 'SPH_lat<a>lon<b>[z<c>]'.

    SPH: lat=0 → +Z (top), lat=180 → -Z (=mz), +lon → CCW from +X toward +Y.
    Optional z suffix spins around the approach axis.
    """
    if s in _EEF_DIR_ROT:
        return _EEF_DIR_ROT[s]

    if s.upper().startswith('SPH_'):
        raw   = s[4:].lower()
        lat_m = re.search(r'lat(-?\d+(?:\.\d+)?)', raw)
        lon_m = re.search(r'lon(-?\d+(?:\.\d+)?)', raw)
        z_m   = re.search(r'z(-?\d+(?:\.\d+)?)',   raw)
        if not lat_m or not lon_m:
            raise ValueError(f"Invalid SPH format {s!r}. Use e.g. 'SPH_lat90lon0' or 'SPH_lat90lon0z10'.")
        lat   = float(lat_m.group(1))
        lon   = float(lon_m.group(1))
        z_rot = float(z_m.group(1)) if z_m else 0.0
        # approach dir d = [sin(lat)cos(lon), sin(lat)sin(lon), cos(lat)]; R @ [0,0,-1] = d
        return (Rotation.from_euler('z', lon, degrees=True)
                * Rotation.from_euler('y', lat + 180, degrees=True)
                * Rotation.from_euler('z', z_rot, degrees=True))

    raise ValueError(f"Invalid --eef-dir {s!r}. Use 'mz'/'py'/'my' or 'SPH_lat<a>lon<b>[z<c>]'.")


# ── data loading + transforms ─────────────────────────────────────────────────

def load_traj(data_dir):
    poses = np.load(Path(data_dir) / "object_pose" / "poses.npz")["poses"]
    return poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()


def cam_to_robot_matrix(angle_deg, elevation_deg=0.0, R0=DATASET_CAM_FRAME_IN_ROBOT):
    elev = Rotation.from_euler('x', -elevation_deg, degrees=True).as_matrix()
    return Rotation.from_euler('z', -angle_deg, degrees=True).as_matrix() @ R0 @ elev


def cam_to_robot(pos, quat, R):
    return (R @ pos.T).T, (Rotation.from_matrix(R) * Rotation.from_quat(quat)).as_quat()


def remap(pos, quat, center=(0.0, 0.0, 1.0), scale=1.0):
    return (pos - pos.mean(axis=0)) * scale + np.array(center), quat


def compute_dataset_cam(pos_cam_raw, scale, center, R):
    mean_raw = (R @ pos_cam_raw.T).T.mean(axis=0)
    cam_pos  = (np.zeros(3) - mean_raw) * scale + np.array(center)
    mujoco_cam_mat = np.column_stack([R[:, 0], -R[:, 1], -R[:, 2]])
    q = Rotation.from_matrix(mujoco_cam_mat).as_quat()
    return cam_pos, np.array([q[3], q[0], q[1], q[2]])


def _hide_default_lift_props(root):
    """Hide Lift table and cube without deleting them (robosuite references their names)."""
    hide_prefixes = ("table_", "cube_g0")
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name.startswith(hide_prefixes):
            rgba = geom.get("rgba", "1 1 1 1").split()
            if len(rgba) != 4:
                rgba = ["1", "1", "1", "1"]
            rgba[3] = "0"
            geom.set("rgba", " ".join(rgba))
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


# ── IK ────────────────────────────────────────────────────────────────────────

def solve_ik(env, target_pos, target_rot, n_iter=1000, tol=1e-4, damping=0.01, max_dq=0.2):
    import mujoco
    robot     = env.robots[0]
    m         = getattr(env.sim.model, '_model', env.sim.model)
    d         = getattr(env.sim.data,  '_data',  env.sim.data)
    site_id   = next(iter(robot.eef_site_id.values())) if isinstance(robot.eef_site_id, dict) \
                else int(robot.eef_site_id)
    eef_name  = next(iter(robot.robot_model.eef_name.values())) if isinstance(robot.robot_model.eef_name, dict) \
                else robot.robot_model.eef_name
    body_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, eef_name)
    joint_ids = list(robot._ref_joint_pos_indexes)
    nv        = m.nv
    jacp, jacr, _dummy = np.zeros((3, nv)), np.zeros((3, nv)), np.zeros((3, nv))
    R_target  = target_rot.as_matrix()
    err       = np.ones(6)
    for _ in range(n_iter):
        mujoco.mj_forward(m, d)
        dp  = target_pos - d.site_xpos[site_id]
        dR  = Rotation.from_matrix(R_target @ d.xmat[body_id].reshape(3, 3).T).as_rotvec()
        err = np.concatenate([dp, dR])
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(m, d, jacp,   _dummy, site_id)
        mujoco.mj_jacBody(m, d, _dummy, jacr,   body_id)
        J  = np.vstack([jacp, jacr])[:, joint_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
        d.qpos[joint_ids] += np.clip(dq, -max_dq, max_dq)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    print(f"IK residual: {np.linalg.norm(err):.5f}")


# ── simulation ────────────────────────────────────────────────────────────────

def run_sim(pos, quat, pos_cam_raw, scale, center,
            steps_per_waypoint, video_dir, run_name, wandb_run,
            R=DATASET_CAM_FRAME_IN_ROBOT, angle=0.0, elevation=0.0, eef_dir='mz',
            show_eef=False, fovy=60.0, cam_h=480, cam_w=640, data_dir=None,
            mesh_rot_offset=None, control_freq=20):

    cam_pos, cam_quat = compute_dataset_cam(pos_cam_raw, scale, center, R)
    dataset_cam_key   = f"dataset_cam_{angle:g}_elev{elevation:g}_{eef_dir}"

    env = suite.make(
        env_name="Lift", robots="PandaRotatedGripper",
        gripper_types="Robotiq85Gripper",
        controller_configs=_load_ctrl_cfg(),
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMERAS, camera_heights=cam_h, camera_widths=cam_w,
        ignore_done=True, horizon=steps_per_waypoint * len(pos) + 100,
        hard_reset=False, control_freq=control_freq,
    )

    # Inject dataset camera and object mesh, then recompile
    root = ET.fromstring(env.model.get_xml())
    _hide_default_lift_props(root)
    ET.SubElement(root.find('worldbody'), 'camera', {
        'name': DATASET_CAM,
        'pos':  ' '.join(f"{v:.4f}" for v in cam_pos),
        'quat': ' '.join(f"{v:.6f}" for v in cam_quat),
        'fovy': f'{fovy:.4f}',
    })
    if data_dir is not None:
        inject_object_xml(root, data_dir)
    env._initialize_sim(ET.tostring(root, encoding='unicode'))
    env.reset()
    _set_droid_init_qpos(env)
    obs = env._get_observations()

    # Read EEF state directly from MuJoCo (obs may be stale after _set_droid_init_qpos)
    _eef_body_name = next(iter(env.robots[0].robot_model.eef_name.values())) if isinstance(env.robots[0].robot_model.eef_name, dict) else env.robots[0].robot_model.eef_name
    _body_id = env.sim.model.body_name2id(_eef_body_name)
    _site_id = next(iter(env.robots[0].eef_site_id.values())) if isinstance(env.robots[0].eef_site_id, dict) else int(env.robots[0].eef_site_id)

    # Orient gripper to match trajectory start, then IK to first waypoint
    rot_eef_init      = Rotation.from_matrix(env.sim.data.body_xmat[_body_id].reshape(3, 3))
    rot_dataset_first = Rotation.from_quat(quat[0])
    rot_first_ik = _parse_eef_dir(eef_dir) * rot_eef_init

    solve_ik(env, pos[0], rot_first_ik)
    obs = env._get_observations()

    # Read site orientation directly from sim data after IK (obs may cache pre-IK state)
    rot_first = Rotation.from_matrix(env.sim.data.site_xmat[_site_id].reshape(3, 3))
    C_inv     = rot_first.inv() * rot_dataset_first
    
    # Rigidly attach object to EEF (grip_site == pos[0] after IK converges)
    quat0_mesh = quat[0]
    if mesh_rot_offset is not None:
        quat0_mesh = (Rotation.from_quat(quat[0]) * mesh_rot_offset.inv()).as_quat()
    update_object = make_object_updater(env, quat0_mesh) if data_dir is not None else None
    if update_object:
        update_object()

    cam_mat     = {cam: get_cam_matrices(env, cam, cam_h, cam_w) for cam in CAMERAS + [DATASET_CAM]}
    frames      = {cam: [] for cam in CAMERAS + [dataset_cam_key]}
    eef_history = []

    for i, (tgt_pos, tgt_quat) in enumerate(zip(pos, quat)):
        rot_target = Rotation.from_quat(tgt_quat) * rot_dataset_first.inv() * rot_first
        action = np.concatenate([tgt_pos, rot_target.as_rotvec(), [1.0]])
        for _ in range(steps_per_waypoint):
            obs, *_ = env.step(action)

            if update_object:
                update_object()

            eef_history.append(obs["robot0_eef_pos"].copy())
            eef_quat_vis = (Rotation.from_quat(obs["robot0_eef_quat_site"]) * C_inv).as_quat()

            for cam in CAMERAS:
                img = env.sim.render(cam_w, cam_h, camera_name=cam)[::-1].copy()
                if show_eef:
                    draw_eef(img, eef_history, eef_quat_vis, *cam_mat[cam])
                frames[cam].append(img)

            ds_img = env.sim.render(cam_w, cam_h, camera_name=DATASET_CAM)[::-1].copy()
            if show_eef:
                draw_eef(ds_img, eef_history, eef_quat_vis, *cam_mat[DATASET_CAM])
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


def plot_traj(pos_cam, quat_cam, pos_robot, quat_robot, video_dir, run_name):
    """Save two 6-panel trajectory plots (camera frame and robot frame)."""
    out_dir = Path(video_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t          = np.arange(len(pos_cam))
    rpy_cam    = Rotation.from_quat(quat_cam).as_euler('xyz', degrees=True)
    rpy_robot  = Rotation.from_quat(quat_robot).as_euler('xyz', degrees=True)

    datasets = [
        ("Camera Frame (original)",         pos_cam,   rpy_cam,   "traj_cam.png"),
        ("Robot Frame (cam→robot + remap)",  pos_robot, rpy_robot, "traj_robot.png"),
    ]
    dim_labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    units      = ['m'] * 3 + ['deg'] * 3
    colors     = ['#e74c3c', '#2ecc71', '#3498db', '#e67e22', '#9b59b6', '#1abc9c']

    for title, pos, rpy, fname in datasets:
        data = np.concatenate([pos, rpy], axis=1)   # (N, 6)
        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(title, fontsize=13, fontweight='bold')
        for ax, d, label, unit, color in zip(axes, data.T, dim_labels, units, colors):
            ax.plot(t, d, color=color, linewidth=1.2)
            ax.set_ylabel(f'{label} ({unit})', fontsize=9)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('frame index')
        fig.tight_layout()
        path = str(out_dir / fname)
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot  -> {path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = _load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",    nargs="?", default=cfg.get("data_dir", "data/035_power_drill_20200709_151335"))
    parser.add_argument("--scale",     type=float, default=cfg.get("scale", 1))
    parser.add_argument("--steps",     type=int,   default=cfg.get("steps", 2))
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--project",   default="robosuite-eef-traj")
    parser.add_argument("--name",      default=None)
    parser.add_argument("--wandb",     action="store_true")
    parser.add_argument("--show-eef",  action="store_true", default=cfg.get("show_eef", False))
    parser.add_argument("--angle",     type=float, default=cfg.get("angle", 90))
    parser.add_argument("--elevation", type=float, default=cfg.get("elevation", 0.0),
                        help="vertical camera angle in degrees (0=horizontal, 90=top-down)")
    parser.add_argument("--eef-dir",   default=cfg.get("eef_dir", "mz"),
                        help="gripper approach: 'mz'/'py'/'my' or 'SPH_lat<a>lon<b>[z<c>]' (e.g. 'SPH_lat180lon0' for top-down)")
    parser.add_argument("--ref-dir",   default=None,
                        help="reference data dir (e.g. data/ours) whose frame 0 defines the canonical mesh orientation")
    parser.add_argument("--control-freq", type=float, default=cfg.get("control_freq", 10),
                        help="OSC control frequency in Hz (default: 20)")
    args = parser.parse_args()

    _tmp_env = suite.make(
        env_name="Lift", robots="PandaRotatedGripper", gripper_types="Robotiq85Gripper",
        controller_configs=_load_ctrl_cfg(),
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, horizon=10,
    )
    _tmp_env.reset()
    _set_droid_init_qpos(_tmp_env)
    _tmp_robot = _tmp_env.robots[0]
    _tmp_site_id = next(iter(_tmp_robot.eef_site_id.values())) if isinstance(_tmp_robot.eef_site_id, dict) else int(_tmp_robot.eef_site_id)
    center = tuple(_tmp_env.sim.data.site_xpos[_tmp_site_id].copy())
    # _init_img = _tmp_env.sim.render(640, 480, camera_name="sideview")[::-1]
    # imageio.imwrite(str(PROJECT_ROOT / "init_pose_sideview.png"), _init_img)
    # print(f"Saved initial pose image -> {PROJECT_ROOT / 'init_pose_sideview.png'}")
    _tmp_env.close()
    
    center = tuple(np.array(center) + np.array(cfg.get("center_offset", [-0.1, 0.0, 0.1])))    
    
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    run_name  = args.name or data_dir.name
    R         = cam_to_robot_matrix(args.angle, args.elevation)

    pos_cam, quat_cam = load_traj(data_dir)
    pos, quat = cam_to_robot(pos_cam, quat_cam, R)
    pos, quat = remap(pos, quat, center=center, scale=args.scale)

    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = PROJECT_ROOT / video_dir

    cam_json = json.load(open(data_dir / "camera.json"))
    cam_h    = cam_json["height"]
    cam_w    = cam_json["width"]
    fovy     = math.degrees(2 * math.atan(cam_h / (2 * cam_json["intrinsics"][1][1])))

    make_video(data_dir)
    plot_traj(pos_cam, quat_cam, pos, quat, video_dir, run_name)

    # mesh orientation correction: align frame-0 body frame to reference canonical pose
    mesh_rot_offset = None
    if args.ref_dir is not None:
        ref_dir = Path(args.ref_dir)
        if not ref_dir.is_absolute():
            ref_dir = PROJECT_ROOT / ref_dir
        R_ref  = np.load(ref_dir  / "object_pose" / "poses.npz")["poses"][0, :3, :3].astype(float)
        R_cur  = np.load(data_dir / "object_pose" / "poses.npz")["poses"][0, :3, :3].astype(float)
        mesh_rot_offset = Rotation.from_matrix(R_ref.T @ R_cur)

    wandb_run = wandb.init(project=args.project, name=run_name) if args.wandb else None
    run_sim(pos, quat, pos_cam, args.scale, center,
            args.steps, video_dir, run_name, wandb_run, R=R,
            angle=args.angle, elevation=args.elevation, eef_dir=args.eef_dir, show_eef=args.show_eef,
            fovy=fovy, cam_h=cam_h, cam_w=cam_w, data_dir=data_dir,
            mesh_rot_offset=mesh_rot_offset, control_freq=args.control_freq)
    if wandb_run:
        wandb_run.finish()
