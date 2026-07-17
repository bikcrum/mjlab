"""Frame transformer sensor demo.

A YAM robot arm is mounted on a lit, checkered ground plane. The
FrameTransformer sensor tracks the arm's grasp site (end effector) and wrist
body relative to the arm's fixed base as the arm sweeps through a joint-space
motion. With ``debug_vis=True`` the viewer draws a coordinate frame at the
source (base) and at each target, plus a connector line from the source to
each target, so the sensor's relative-transform readings can be checked
visually as the arm moves.

Run with:
  uv run mjpython scripts/demos/frame_transformer_sensor.py                # macOS
  uv run python scripts/demos/frame_transformer_sensor.py                  # Linux
  uv run python scripts/demos/frame_transformer_sensor.py --viewer viser   # Viser
"""

from __future__ import annotations

import dataclasses
import math
import os

import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots.i2rt_yam.yam_constants import get_yam_robot_cfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sensor import FrameCfg, FrameTransformerCfg, OffsetCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# Where the arm's base is spawned, away from the world origin and up in the
# air so the sweep doesn't drive the arm's links into the ground plane.
BASE_POS = (0.5, 0.5, 0.5)

# Joints animated during the sweep: (name, center, amplitude, angular_rate).
# Ranges: joint1 [-2.62, 3.05], joint2/3 [0, 3.67], joint4/5 [-1.57, 1.57],
# joint6 [-2.09, 2.09]. Amplitudes stay well inside each joint's travel to
# avoid self-collision (extremes on joint2/joint3 together fold the arm into
# itself and overflow the contact-constraint buffer).
_ARM_SWEEP = (
  ("joint1", 0.0, 1.2, 0.6),
  ("joint2", 1.6, 1.0, 0.8),
  ("joint3", 1.6, 1.0, 0.7),
  ("joint4", 0.0, 0.9, 0.9),
  ("joint5", 0.0, 0.9, 1.0),
  ("joint6", 0.0, 1.2, 1.2),
)
_TIME_SCALE = 0.02  # Faster oscillation than the raw per-step counter.
_GRIPPER_HOLD = {"left_finger": 0.01875, "right_finger": -0.01875}

# How often (in policy steps) to print target poses relative to the source.
_PRINT_EVERY = 50


def create_env_cfg() -> ManagerBasedRlEnvCfg:
  robot_cfg = get_yam_robot_cfg()
  robot_cfg = dataclasses.replace(
    robot_cfg,
    init_state=dataclasses.replace(robot_cfg.init_state, pos=BASE_POS),
  )

  frame_cfg = FrameTransformerCfg(
    name="ee_frame",
    entity="robot",
    source_body_name="arm",
    target_frames=[
      FrameCfg(
        body_name="grasp_site",
        obj_type="site",
        name="end_effector",
        offset=OffsetCfg(pos=(0.0, 0.0, 0.02)),
      ),
      FrameCfg(body_name="link_6", name="wrist"),
    ],
    debug_vis=True,
  )

  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    scene=SceneCfg(
      num_envs=1,
      env_spacing=0.0,
      extent=2.0,
      terrain=TerrainEntityCfg(terrain_type="plane", num_envs=1),
      entities={"robot": robot_cfg},
      sensors=(frame_cfg,),
    ),
  )

  cfg.viewer.body_name = "arm"
  cfg.viewer.distance = 1.2
  cfg.viewer.elevation = -20.0

  return cfg


class ArmSweepPolicy:
  """Drives the arm through a smooth joint-space sweep each step."""

  def __init__(self, env, device: str) -> None:
    self._env = env
    self._device = device
    self._step_count = 0

    robot = env.unwrapped.scene["robot"]
    joint_names = robot.joint_names
    self._robot = robot
    self._sensor = env.unwrapped.scene["robot/ee_frame"]
    self._arm_joint_ids = torch.tensor(
      [joint_names.index(name) for name, *_ in _ARM_SWEEP],
      device=device,
      dtype=torch.long,
    )
    self._gripper_joint_ids = torch.tensor(
      [joint_names.index(name) for name in _GRIPPER_HOLD],
      device=device,
      dtype=torch.long,
    )
    self._gripper_target = torch.tensor(
      list(_GRIPPER_HOLD.values()), device=device, dtype=torch.float32
    ).unsqueeze(0)

  def __call__(self, obs: object) -> torch.Tensor:
    del obs
    t = self._step_count * _TIME_SCALE

    arm_target = torch.tensor(
      [
        center + amplitude * math.sin(rate * t)
        for _, center, amplitude, rate in _ARM_SWEEP
      ],
      device=self._device,
      dtype=torch.float32,
    ).unsqueeze(0)

    self._robot.set_joint_position_target(arm_target, joint_ids=self._arm_joint_ids)
    self._robot.set_joint_position_target(
      self._gripper_target, joint_ids=self._gripper_joint_ids
    )

    if self._step_count % _PRINT_EVERY == 0:
      self._print_target_poses()

    self._step_count += 1
    return torch.zeros(self._env.unwrapped.action_space.shape, device=self._device)

  def _print_target_poses(self) -> None:
    data = self._sensor.data
    print(f"--- step {self._step_count} ---")
    for i, name in enumerate(self._sensor.target_frame_names):
      pos = data.target_pos_source[0, i].tolist()
      quat = data.target_quat_source[0, i].tolist()
      pos_str = ", ".join(f"{v:+.3f}" for v in pos)
      quat_str = ", ".join(f"{v:+.3f}" for v in quat)
      print(f"  {name:>13}: pos=({pos_str})  quat=({quat_str})")


def main(device: str = "cpu", viewer: str = "auto") -> None:
  configure_torch_backends()

  env_cfg = create_env_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  print("=" * 50)
  print("Frame Transformer Sensor Demo (YAM arm)")
  print("  source: arm (fixed base)")
  print("  end_effector: grasp_site (site on link_6)")
  print("  wrist: link_6 body")
  print("=" * 50)

  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = viewer

  policy = ArmSweepPolicy(env, device)
  if resolved == "native":
    print("Launching native viewer...")
    NativeMujocoViewer(env, policy).run()
  elif resolved == "viser":
    print("Launching Viser viewer...")
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer: {viewer}")

  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
