# object_traj

<table>
  <tr>
    <td align="center"><b>Demo</b></td>
    <td align="center"><b>Robot Tracking (same viewpoint)</b></td>
  </tr>
  <tr>
    <td><img src="document/trajectory.gif" width="360"/></td>
    <td><img src="document/dataset_cam_50_py_origin.gif" width="360"/></td>
  </tr>
</table>

## Setup

```bash
uv sync
source .venv/bin/activate
```

Place dataset folders inside `data/`:

```
data/
  011_banana_20200709_145401/
  035_power_drill_20200709_151335/
  ...
```

---

## `src/main.py` — Robosuite Simulation

Loads an object trajectory from a dataset, converts it from camera frame to robot frame,
and replays it in a robosuite (Panda) simulation using OSC control.
Also generates a dataset trajectory visualization (`trajectory.mp4`) before running the simulation.
Records video from multiple camera views (front, bird, side, dataset_cam) and optionally logs to wandb.

**Options:**
- `--show-eef`: overlay EEF trail and orientation axes on all camera views
- `--angle`: horizontal camera angle in degrees (0=head-on, +90=robot's right, -90=robot's left)
- `--eef-dir`: initial gripper orientation of the end-effector (my=-y dir, mz=-z dir, py=+y dir)
- `--scale`: scale factor for trajectory size in robot space (default: 1.0)
- `--steps`: simulation steps per waypoint (default: 2; fewer steps = faster but larger error)
- `--no-wandb`: skip wandb logging

```bash
python src/main.py --no-wandb
python src/main.py data/011_banana_20200709_145401 --no-wandb
python src/main.py data/011_banana_20200709_145401 --angle 45 --eef-dir my --show-eef --no-wandb
python src/main.py data/011_banana_20200709_145401 --steps 20 --scale 2.0
python src/main.py data/011_banana_20200709_145401 --video-dir videos --project my-project --name my-run
```

Output:
- `videos/<run_name>/{frontview,birdview,sideview,dataset_cam}.mp4` — simulation camera views
- `videos/<run_name>/trajectory.mp4` — original dataset frames with projected object trajectory overlaid

`dataset_cam_<angle>_<gripper>.mp4` is rendered from the same point of view as the camera used to record the original dataset, rotated by `--angle` degrees around the robot.

---

## Multi-angle Comparison

The `--angle` flag controls where the camera is placed relative to the robot. Below are example results from three viewpoints:

<table>
  <tr>
    <td align="center"><b>--angle -50</b></td>
    <td align="center"><b>--angle 0</b></td>
    <td align="center"><b>--angle 50</b></td>
  </tr>
  <tr>
    <td><img src="document/dataset_cam_-50_py.gif" width="240"/></td>
    <td><img src="document/dataset_cam_0_py.gif" width="240"/></td>
    <td><img src="document/dataset_cam_50_py_m.gif" width="240"/></td>
  </tr>
</table>

## Gripper multi-pose Comparison

The `--eef-dir` flag controls the initial gripper orientation of the end-effector:

<table>
  <tr>
    <td align="center"><b>--eef-dir my</b></td>
    <td align="center"><b>--eef-dir mz (default)</b></td>
    <td align="center"><b>--eef-dir py</b></td>
  </tr>
  <tr>
    <td><img src="document/dataset_cam_50_my.gif" width="240"/></td>
    <td><img src="document/dataset_cam_50_mz.gif" width="240"/></td>
    <td><img src="document/dataset_cam_50_py.gif" width="240"/></td>
  </tr>
</table>

`--eef-dir` also provides a gripper orientation control function based on spherical coordinates. The direction the gripper fingers point can be specified using latitude and longitude in spherical coordinates. After setting the latitude and longitude, the yaw rotation can also be specified.

<table>
  <tr>
    <td align="center"><b>--eef-dir SPH_lat150lon0z0</b></td>
    <td align="center"><b>--eef-dir SPH_lat150lon30z0</b></td>
    <td align="center"><b>--eef-dir SPH_lat180lon0z0</b></td>
    <td align="center"><b>--eef-dir SPH_lat180lon0z90</b></td>
  </tr>
  <tr>
    <td><img src="document/dataset_cam_50_SPH_lat150lon0z0.gif" width="240"/></td>
    <td><img src="document/dataset_cam_50_SPH_lat150lon30z0.gif" width="240"/></td>
    <td><img src="document/dataset_cam_50_SPH_lat180lon0z0.gif" width="240"/></td>
    <td><img src="document/dataset_cam_50_SPH_lat180lon0z90.gif" width="240"/></td>
  </tr>
</table>