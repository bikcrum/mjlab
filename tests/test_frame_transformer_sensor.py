"""Tests for the FrameTransformer sensor."""

from __future__ import annotations

import pytest
import torch
from conftest import get_test_device, make_scene_and_sim

from mjlab.sensor import (
  FrameCfg,
  FrameObjType,
  FrameTransformer,
  FrameTransformerCfg,
  OffsetCfg,
)
from mjlab.utils.lab_api.math import quat_from_angle_axis

# Base body at (0, 0, 1), no rotation. "arm" body offset (1, 0, 0) from base,
# "head" body offset (0, 0, 0.5) from base. All frames axis-aligned with world.
TWO_BODY_XML = """
  <mujoco>
    <worldbody>
      <body name="base" pos="0 0 1">
        <freejoint name="free_joint"/>
        <geom name="base_geom" type="sphere" size="0.05" mass="1.0"/>
        <body name="arm" pos="1 0 0">
          <geom name="arm_geom" type="sphere" size="0.05" mass="1.0"/>
          <site name="arm_site" pos="0 0 0.2"/>
        </body>
        <body name="head" pos="0 0 0.5">
          <geom name="head_geom" type="sphere" size="0.05" mass="1.0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
"""

# 132 deg about the (1, 1, 1) axis: a generic rotation that mixes all three
# world axes, so the test can't accidentally pass via axis-aligned symmetry.
_GENERIC_ROT = quat_from_angle_axis(
  torch.tensor([2.3]), torch.tensor([[1.0, 1.0, 1.0]])
)[0].tolist()

# Base body has a non-identity, non-axis-aligned world orientation. Child
# "arm" is offset (1, 0, 0) in the base's *local* frame with no additional
# rotation of its own, so its world orientation equals the base's exactly.
ROTATED_BASE_XML = f"""
  <mujoco>
    <worldbody>
      <body name="base" pos="0 0 1"
            quat="{_GENERIC_ROT[0]} {_GENERIC_ROT[1]} {_GENERIC_ROT[2]} {_GENERIC_ROT[3]}">
        <freejoint name="free_joint"/>
        <geom name="base_geom" type="sphere" size="0.05" mass="1.0"/>
        <body name="arm" pos="1 0 0">
          <geom name="arm_geom" type="sphere" size="0.05" mass="1.0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _cfg(
  target_frames: list[FrameCfg] | None = None,
  source_body_name: str = "base",
  source_obj_type: FrameObjType | None = None,
  obj_type: FrameObjType = "xbody",
  source_frame_offset: OffsetCfg | None = None,
) -> FrameTransformerCfg:
  return FrameTransformerCfg(
    name="frame_xform",
    entity="robot",
    source_body_name=source_body_name,
    source_obj_type=source_obj_type,
    obj_type=obj_type,
    target_frames=target_frames
    if target_frames is not None
    else [FrameCfg(body_name="arm"), FrameCfg(body_name="head")],
    source_frame_offset=source_frame_offset
    if source_frame_offset is not None
    else OffsetCfg(),
  )


def test_target_pos_source_matches_static_offsets(device):
  """Target positions relative to the source should match the known offsets."""
  scene, sim = make_scene_and_sim(device, TWO_BODY_XML, (_cfg(),))
  sim.step()
  sim.sense()

  sensor: FrameTransformer = scene["robot/frame_xform"]
  data = sensor.data

  assert sensor.target_frame_names == ["arm", "head"]
  assert data.target_pos_source.shape == (1, 2, 3)
  assert data.target_pos_source[0, 0].cpu().tolist() == pytest.approx(
    [1.0, 0.0, 0.0], abs=1e-4
  )
  assert data.target_pos_source[0, 1].cpu().tolist() == pytest.approx(
    [0.0, 0.0, 0.5], abs=1e-4
  )


def test_target_pos_w_matches_world_positions(device):
  """World-frame target positions should reflect the body's world pose."""
  scene, sim = make_scene_and_sim(device, TWO_BODY_XML, (_cfg(),))
  sim.step()
  sim.sense()

  data = scene["robot/frame_xform"].data
  assert data.target_pos_w[0, 0].cpu().tolist() == pytest.approx(
    [1.0, 0.0, 1.0], abs=1e-4
  )
  assert data.source_pos_w[0].cpu().tolist() == pytest.approx([0.0, 0.0, 1.0], abs=1e-4)


def test_default_name_falls_back_to_body_name(device):
  """A target frame without an explicit name uses its body_name."""
  cfg = _cfg(target_frames=[FrameCfg(body_name="arm", name="end_effector")])
  scene, sim = make_scene_and_sim(device, TWO_BODY_XML, (cfg,))
  sim.step()
  sim.sense()

  assert scene["robot/frame_xform"].target_frame_names == ["end_effector"]


def test_site_target_frame(device):
  """Target frames can reference sites, not just bodies."""
  cfg = _cfg(
    target_frames=[FrameCfg(body_name="arm_site", obj_type="site")],
    obj_type="site",
    source_body_name="base",
    source_obj_type="xbody",
  )
  scene, sim = make_scene_and_sim(device, TWO_BODY_XML, (cfg,))
  sim.step()
  sim.sense()

  data = scene["robot/frame_xform"].data
  # arm_site is offset (0, 0, 0.2) from arm, which is (1, 0, 0) from base.
  assert data.target_pos_source[0, 0].cpu().tolist() == pytest.approx(
    [1.0, 0.0, 0.2], abs=1e-4
  )


def test_source_offset_applied(device):
  """A non-identity source offset shifts every target-relative reading."""
  cfg = _cfg(source_frame_offset=OffsetCfg(pos=(0.0, 0.0, 1.0)))
  scene, sim = make_scene_and_sim(device, TWO_BODY_XML, (cfg,))
  sim.step()
  sim.sense()

  data = scene["robot/frame_xform"].data
  # Source frame moves up by 1.0, so the arm now reads 1.0 lower in z.
  assert data.target_pos_source[0, 0].cpu().tolist() == pytest.approx(
    [1.0, 0.0, -1.0], abs=1e-4
  )


def test_target_offset_rotation_applied(device):
  """A non-identity target offset rotation is reflected in the world/relative
  orientation, not just position."""
  offset_rot = quat_from_angle_axis(
    torch.tensor([1.2]), torch.tensor([[0.0, 0.0, 1.0]])
  )[0].tolist()
  cfg = _cfg(
    target_frames=[
      FrameCfg(body_name="arm", offset=OffsetCfg(rot=tuple(offset_rot))),
      FrameCfg(body_name="head"),
    ]
  )
  scene, sim = make_scene_and_sim(device, TWO_BODY_XML, (cfg,))
  sim.step()
  sim.sense()

  data = scene["robot/frame_xform"].data
  # "arm" and "base" both have identity world orientation, so the offset
  # rotation passes through unchanged into both the world and source frames.
  assert data.target_quat_w[0, 0].cpu().tolist() == pytest.approx(offset_rot, abs=1e-4)
  assert data.target_quat_source[0, 0].cpu().tolist() == pytest.approx(
    offset_rot, abs=1e-4
  )
  # "head" has no offset, so it keeps identity orientation.
  assert data.target_quat_w[0, 1].cpu().tolist() == pytest.approx(
    [1.0, 0.0, 0.0, 0.0], abs=1e-4
  )


def test_source_rotation_cancels_in_target_pos_source(device):
  """A rotated source frame doesn't leak into target_pos_source/quat_source.

  "arm" is offset (1, 0, 0) in the base's *local* frame with no rotation of
  its own, so regardless of the base's world orientation, the target-relative
  reading must exactly equal that local offset, and the relative orientation
  must stay identity.
  """
  cfg = FrameTransformerCfg(
    name="frame_xform",
    entity="robot",
    source_body_name="base",
    target_frames=[FrameCfg(body_name="arm")],
  )
  scene, sim = make_scene_and_sim(device, ROTATED_BASE_XML, (cfg,))
  sim.step()
  sim.sense()

  data = scene["robot/frame_xform"].data
  assert data.source_quat_w[0].cpu().tolist() == pytest.approx(_GENERIC_ROT, abs=1e-4)
  assert data.target_quat_w[0, 0].cpu().tolist() == pytest.approx(
    _GENERIC_ROT, abs=1e-4
  )
  assert data.target_pos_source[0, 0].cpu().tolist() == pytest.approx(
    [1.0, 0.0, 0.0], abs=1e-4
  )
  assert data.target_quat_source[0, 0].cpu().tolist() == pytest.approx(
    [1.0, 0.0, 0.0, 0.0], abs=1e-4
  )


def test_cross_entity_source_and_target(device):
  """Source and target frames may live in different entities.

  With ``entity=None``, body names are resolved literally, so a
  fully-qualified ``"{entity}/{body}"`` name can point at a body outside the
  source's own entity.
  """
  base_xml = """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.05" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
  """
  target_xml = """
    <mujoco>
      <worldbody>
        <body name="target" pos="2 0 1">
          <freejoint name="free_joint"/>
          <geom name="target_geom" type="sphere" size="0.05" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
  """
  cfg = FrameTransformerCfg(
    name="frame_xform",
    source_body_name="base_entity/base",
    target_frames=[FrameCfg(body_name="target_entity/target", name="target")],
  )
  scene, sim = make_scene_and_sim(
    device, {"base_entity": base_xml, "target_entity": target_xml}, (cfg,)
  )
  sim.step()
  sim.sense()

  sensor: FrameTransformer = scene["frame_xform"]
  data = sensor.data
  assert sensor.target_frame_names == ["target"]
  assert data.target_pos_source[0, 0].cpu().tolist() == pytest.approx(
    [2.0, 0.0, 0.0], abs=1e-4
  )


def test_target_offset_applied(device):
  """A non-identity target offset shifts that target's world/relative pose."""
  cfg = _cfg(
    target_frames=[
      FrameCfg(body_name="arm", offset=OffsetCfg(pos=(0.0, 0.0, 0.3))),
      FrameCfg(body_name="head"),
    ]
  )
  scene, sim = make_scene_and_sim(device, TWO_BODY_XML, (cfg,))
  sim.step()
  sim.sense()

  data = scene["robot/frame_xform"].data
  assert data.target_pos_source[0, 0].cpu().tolist() == pytest.approx(
    [1.0, 0.0, 0.3], abs=1e-4
  )


def test_duplicate_target_bodies_deduplicate_sensors(device):
  """Two target frames on the same body should share the underlying sensors."""
  cfg = _cfg(
    target_frames=[
      FrameCfg(body_name="arm", name="arm_a"),
      FrameCfg(body_name="arm", name="arm_b", offset=OffsetCfg(pos=(0.0, 0.0, 1.0))),
    ]
  )
  scene, sim = make_scene_and_sim(device, TWO_BODY_XML, (cfg,))
  sim.step()
  sim.sense()

  sensor: FrameTransformer = scene["robot/frame_xform"]
  data = sensor.data
  assert sensor.target_frame_names == ["arm_a", "arm_b"]
  assert data.target_pos_source[0, 0].cpu().tolist() == pytest.approx(
    [1.0, 0.0, 0.0], abs=1e-4
  )
  assert data.target_pos_source[0, 1].cpu().tolist() == pytest.approx(
    [1.0, 0.0, 1.0], abs=1e-4
  )


def test_multi_env(device):
  """Sensor reports identical readings across identical environments."""
  scene, sim = make_scene_and_sim(device, TWO_BODY_XML, (_cfg(),), num_envs=4)
  sim.step()
  sim.sense()

  data = scene["robot/frame_xform"].data
  assert data.target_pos_source.shape == (4, 2, 3)
  for i in range(4):
    assert data.target_pos_source[i, 0].cpu().tolist() == pytest.approx(
      [1.0, 0.0, 0.0], abs=1e-4
    )


def test_requires_at_least_one_target_frame():
  """FrameTransformerCfg rejects an empty target_frames list."""
  with pytest.raises(ValueError, match="target_frames"):
    FrameTransformerCfg(name="frame_xform", source_body_name="base", target_frames=[])


def test_requires_source_body_name():
  """FrameTransformerCfg rejects an empty source_body_name."""
  with pytest.raises(ValueError, match="source_body_name"):
    FrameTransformerCfg(
      name="frame_xform",
      source_body_name="",
      target_frames=[FrameCfg(body_name="arm")],
    )


def test_invalid_obj_type_rejected():
  """FrameTransformerCfg rejects an obj_type outside the spatial frame set."""
  with pytest.raises(ValueError, match="obj_type"):
    FrameTransformerCfg(
      name="frame_xform",
      source_body_name="base",
      target_frames=[FrameCfg(body_name="arm")],
      obj_type="joint",  # type: ignore[invalid-argument-type]
    )
