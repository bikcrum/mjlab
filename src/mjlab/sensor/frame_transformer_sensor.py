"""Frame transformer sensor.

Reports the transform of one or more target frames with respect to a source
frame. The source and target frames are specified by body (or xbody/site/geom/
camera) names, with optional per-frame offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.sensor.builtin_sensor import (
  _OBJECT_TYPE_MAP,
  _SENSOR_TYPE_MAP,
  _SPATIAL_FRAME_TYPES,
)
from mjlab.sensor.sensor import Sensor, SensorCfg
from mjlab.utils.lab_api.math import (
  combine_frame_transforms,
  is_identity_pose,
  matrix_from_quat,
  subtract_frame_transforms,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.viewer.debug_visualizer import DebugVisualizer


FrameObjType = Literal["body", "xbody", "geom", "site", "camera"]


@dataclass
class OffsetCfg:
  """The offset pose of one frame relative to another frame."""

  pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Translation w.r.t. the parent frame."""

  rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  """Quaternion (w, x, y, z) w.r.t. the parent frame."""


@dataclass
class FrameCfg:
  """Configuration for a single target coordinate frame."""

  body_name: str
  """Name of the MuJoCo object the frame is attached to.

  Resolved within :attr:`FrameTransformerCfg.entity` if that field is set,
  otherwise treated as a literal MuJoCo name.
  """

  name: str | None = None
  """User-defined frame name. Defaults to :attr:`body_name` when ``None``."""

  offset: OffsetCfg = field(default_factory=OffsetCfg)
  """Pose offset from the parent body frame."""

  obj_type: FrameObjType | None = None
  """Optional MuJoCo object type override for this target frame.

  When ``None``, :attr:`FrameTransformerCfg.obj_type` is used.
  """


@dataclass
class FrameTransformerData:
  """Data container for the frame transformer sensor."""

  target_frame_names: list[str]
  """Target frame names in the order returned by the data tensors."""

  target_pos_source: torch.Tensor
  """Target frame positions w.r.t. the source frame. Shape ``[N, M, 3]``."""

  target_quat_source: torch.Tensor
  """Target frame orientations (w, x, y, z) w.r.t. the source frame.
  Shape ``[N, M, 4]``."""

  target_pos_w: torch.Tensor
  """Target frame positions in the world frame after offsets. Shape ``[N, M, 3]``."""

  target_quat_w: torch.Tensor
  """Target frame orientations (w, x, y, z) in the world frame after offsets.
  Shape ``[N, M, 4]``."""

  source_pos_w: torch.Tensor
  """Source frame position in the world frame after offset. Shape ``[N, 3]``."""

  source_quat_w: torch.Tensor
  """Source frame orientation (w, x, y, z) in the world frame after offset.
  Shape ``[N, 4]``."""


@dataclass
class FrameTransformerCfg(SensorCfg):
  """Configuration for the frame transformer sensor."""

  source_body_name: str = ""
  """Name of the source body (resolved within :attr:`entity` if provided)."""

  target_frames: list[FrameCfg] = field(default_factory=list)
  """List of target frames to track."""

  source_frame_offset: OffsetCfg = field(default_factory=OffsetCfg)
  """Pose offset from the source body frame."""

  entity: str | None = None
  """Optional entity prefix used to resolve body/site/etc. names.

  When set, every body name passed in :attr:`source_body_name` and
  :attr:`target_frames` is prefixed with ``"{entity}/"`` to address the
  attached MuJoCo object inside the scene spec.
  """

  obj_type: FrameObjType = "xbody"
  """MuJoCo object type used to read frames.

  Defaults to ``"xbody"`` (the body kinematic frame). Use ``"body"`` for
  the inertial frame, or ``"site"`` / ``"geom"`` / ``"camera"`` to track
  those object types instead.
  """

  source_obj_type: FrameObjType | None = None
  """Optional MuJoCo object type override for the source frame.

  When ``None``, :attr:`obj_type` is used.
  """

  debug_vis: bool = False
  """Whether to draw frames and source-to-target lines in the debug viewer."""

  debug_vis_frame_scale: float = 5.0
  """Length of each axis arrow in the frame marker, scaled by viewer meansize."""

  debug_vis_axis_radius: float = 0.2
  """Radius/thickness of each axis arrow in the frame marker, scaled by viewer
  meansize."""

  debug_vis_line_radius: float = 0.05
  """Radius of source-to-target connector cylinders, scaled by viewer meansize."""

  debug_vis_line_color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0)
  """RGBA color of source-to-target connector cylinders."""

  @property
  def prefixed_name(self) -> str:
    if self.entity:
      return f"{self.entity}/{self.name}"
    return self.name

  def __post_init__(self) -> None:
    if self.obj_type not in _SPATIAL_FRAME_TYPES:
      raise ValueError(
        f"FrameTransformerCfg: obj_type must be one of "
        f"{sorted(_SPATIAL_FRAME_TYPES)}, got '{self.obj_type}'"
      )
    if (
      self.source_obj_type is not None
      and self.source_obj_type not in _SPATIAL_FRAME_TYPES
    ):
      raise ValueError(
        f"FrameTransformerCfg: source_obj_type must be one of "
        f"{sorted(_SPATIAL_FRAME_TYPES)}, got '{self.source_obj_type}'"
      )
    if not self.source_body_name:
      raise ValueError("FrameTransformerCfg: source_body_name must be set")
    if not self.target_frames:
      raise ValueError(
        "FrameTransformerCfg: target_frames must contain at least one entry"
      )
    for i, tgt in enumerate(self.target_frames):
      if tgt.obj_type is not None and tgt.obj_type not in _SPATIAL_FRAME_TYPES:
        raise ValueError(
          f"FrameTransformerCfg: target_frames[{i}].obj_type must be one of "
          f"{sorted(_SPATIAL_FRAME_TYPES)}, got '{tgt.obj_type}'"
        )

  def build(self) -> FrameTransformer:
    return FrameTransformer(self)


class FrameTransformer(Sensor[FrameTransformerData]):
  """Reports transforms of target frames relative to a source frame.

  Internally, this sensor adds MuJoCo ``framepos`` and ``framequat`` builtin
  sensors for the source body and each unique target body. Per-frame
  offsets are applied in pure tensor operations, and the relative transform
  of each target w.r.t. the source is computed at access time.
  """

  cfg: FrameTransformerCfg

  def __init__(self, cfg: FrameTransformerCfg) -> None:
    super().__init__()
    self.cfg = cfg

    self._device: str | None = None

    # Per-tracked-body sensordata views (one entry per unique body).
    self._tracked_pos_views: list[torch.Tensor] = []
    self._tracked_quat_views: list[torch.Tensor] = []

    # Names of MuJoCo sensors added to the spec (one (pos, quat) per body).
    self._sensor_names: list[tuple[str, str]] = []

    # Source body's index into the tracked list.
    self._source_index: int = 0

    # For each target frame, index into the tracked list.
    self._duplicate_indices_list: list[int] = []
    self._duplicate_indices: torch.Tensor | None = None

    # Resolved target frame names (in output order).
    self._target_frame_names: list[str] = []

    # Offset tensors (allocated in initialize when applicable).
    self._source_offset_pos: torch.Tensor | None = None
    self._source_offset_quat: torch.Tensor | None = None
    self._target_offset_pos: torch.Tensor | None = None
    self._target_offset_quat: torch.Tensor | None = None

    self._apply_source_offset: bool = False
    self._apply_target_offset: bool = False

  # Public properties.

  @property
  def num_target_frames(self) -> int:
    """Number of target frames reported by this sensor."""
    return len(self._target_frame_names)

  @property
  def target_frame_names(self) -> list[str]:
    """Ordered list of target frame names."""
    return list(self._target_frame_names)

  # Sensor interface.

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, "Entity"]) -> None:
    if self.cfg.entity is not None and self.cfg.entity not in entities:
      raise ValueError(
        f"FrameTransformer '{self.cfg.name}': entity '{self.cfg.entity}' "
        f"not found. Available: {list(entities.keys())}"
      )

    existing_sensor_names = {s.name for s in scene_spec.sensors}

    object_to_index: dict[tuple[FrameObjType, str], int] = {}
    self._sensor_names = []
    self._target_frame_names = []
    self._duplicate_indices_list = []

    # Source frame is always tracked at index 0.
    source_obj_type = (
      self.cfg.source_obj_type
      if self.cfg.source_obj_type is not None
      else self.cfg.obj_type
    )
    source_objname = self._resolve_objname(self.cfg.source_body_name)
    source_key = (source_obj_type, source_objname)
    object_to_index[source_key] = 0
    self._add_frame_sensors(
      scene_spec,
      objtype=source_obj_type,
      objname=source_objname,
      existing_sensor_names=existing_sensor_names,
      slot_idx=0,
    )
    self._source_index = 0

    # Process each target frame.
    for tgt in self.cfg.target_frames:
      tgt_obj_type = tgt.obj_type if tgt.obj_type is not None else self.cfg.obj_type
      tgt_objname = self._resolve_objname(tgt.body_name)
      tgt_key = (tgt_obj_type, tgt_objname)
      if tgt_key not in object_to_index:
        slot_idx = len(object_to_index)
        object_to_index[tgt_key] = slot_idx
        self._add_frame_sensors(
          scene_spec,
          objtype=tgt_obj_type,
          objname=tgt_objname,
          existing_sensor_names=existing_sensor_names,
          slot_idx=slot_idx,
        )

      self._duplicate_indices_list.append(object_to_index[tgt_key])
      frame_name = tgt.name if tgt.name is not None else tgt.body_name
      self._target_frame_names.append(frame_name)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    del model
    self._device = device

    # Cache sensordata views per tracked body.
    self._tracked_pos_views = []
    self._tracked_quat_views = []
    for pos_name, quat_name in self._sensor_names:
      pos_sensor = mj_model.sensor(pos_name)
      pos_start = pos_sensor.adr[0]
      pos_dim = pos_sensor.dim[0]
      self._tracked_pos_views.append(
        data.sensordata[:, pos_start : pos_start + pos_dim]
      )

      quat_sensor = mj_model.sensor(quat_name)
      quat_start = quat_sensor.adr[0]
      quat_dim = quat_sensor.dim[0]
      self._tracked_quat_views.append(
        data.sensordata[:, quat_start : quat_start + quat_dim]
      )

    num_envs = data.sensordata.shape[0]

    # Source-frame offset.
    source_offset_pos = torch.tensor(
      self.cfg.source_frame_offset.pos, device=device, dtype=torch.float32
    )
    source_offset_quat = torch.tensor(
      self.cfg.source_frame_offset.rot, device=device, dtype=torch.float32
    )
    self._apply_source_offset = not is_identity_pose(
      source_offset_pos, source_offset_quat
    )
    if self._apply_source_offset:
      self._source_offset_pos = (
        source_offset_pos.unsqueeze(0).expand(num_envs, 3).contiguous()
      )
      self._source_offset_quat = (
        source_offset_quat.unsqueeze(0).expand(num_envs, 4).contiguous()
      )

    # Target-frame offsets.
    offset_pos_list: list[torch.Tensor] = []
    offset_quat_list: list[torch.Tensor] = []
    any_non_identity = False
    for tgt in self.cfg.target_frames:
      p = torch.tensor(tgt.offset.pos, device=device, dtype=torch.float32)
      q = torch.tensor(tgt.offset.rot, device=device, dtype=torch.float32)
      offset_pos_list.append(p)
      offset_quat_list.append(q)
      if not is_identity_pose(p, q):
        any_non_identity = True

    self._apply_target_offset = any_non_identity
    if self._apply_target_offset:
      # Shape [B, F, 3] and [B, F, 4] (not [F, 3]/[F, 4]) even though every
      # env starts out identical: allocating the batch dim up front (as a
      # real, contiguous, per-env tensor) lets external code -- e.g. a
      # stateful event term tracking a randomized object size -- write
      # env-specific offsets in place later (`sensor._target_offset_pos[env_ids,
      # target_idx] = ...`), without needing any change to `_compute_data()`.
      self._target_offset_pos = (
        torch.stack(offset_pos_list, dim=0)
        .unsqueeze(0)
        .expand(num_envs, -1, -1)
        .contiguous()
      )
      self._target_offset_quat = (
        torch.stack(offset_quat_list, dim=0)
        .unsqueeze(0)
        .expand(num_envs, -1, -1)
        .contiguous()
      )

    # Duplicate-index tensor for fanning unique tracked bodies into target
    # frame slots.
    self._duplicate_indices = torch.tensor(
      self._duplicate_indices_list, device=device, dtype=torch.long
    )

  def _compute_data(self) -> FrameTransformerData:
    if not self._tracked_pos_views:
      raise RuntimeError(f"FrameTransformer '{self.cfg.name}' is not initialized")

    # [B, T, 3] and [B, T, 4] across all unique tracked bodies.
    pos_tracked = torch.stack(self._tracked_pos_views, dim=1)
    quat_tracked = torch.stack(self._tracked_quat_views, dim=1)

    B = pos_tracked.shape[0]
    F = len(self._target_frame_names)

    # Source frame.
    source_pos = pos_tracked[:, self._source_index]
    source_quat = quat_tracked[:, self._source_index]
    if self._apply_source_offset:
      assert self._source_offset_pos is not None
      assert self._source_offset_quat is not None
      source_pos, source_quat = combine_frame_transforms(
        source_pos,
        source_quat,
        self._source_offset_pos,
        self._source_offset_quat,
      )

    # Target frames — duplicate from tracked bodies.
    assert self._duplicate_indices is not None
    target_pos_tracked = pos_tracked[:, self._duplicate_indices]
    target_quat_tracked = quat_tracked[:, self._duplicate_indices]

    if self._apply_target_offset:
      assert self._target_offset_pos is not None
      assert self._target_offset_quat is not None
      # Already [B, F, 3]/[B, F, 4] (see `initialize()`) -- per-env by
      # construction, so external code (e.g. a size-tracking event term)
      # can mutate specific (env, target) slices in place and have it
      # reflected here with no further changes needed.
      target_pos_w, target_quat_w = combine_frame_transforms(
        target_pos_tracked.reshape(-1, 3),
        target_quat_tracked.reshape(-1, 4),
        self._target_offset_pos.reshape(-1, 3),
        self._target_offset_quat.reshape(-1, 4),
      )
      target_pos_w = target_pos_w.view(B, F, 3)
      target_quat_w = target_quat_w.view(B, F, 4)
    else:
      target_pos_w = target_pos_tracked
      target_quat_w = target_quat_tracked

    # Source-relative transform.
    source_pos_exp = source_pos.unsqueeze(1).expand(-1, F, -1).reshape(-1, 3)
    source_quat_exp = source_quat.unsqueeze(1).expand(-1, F, -1).reshape(-1, 4)
    target_pos_source, target_quat_source = subtract_frame_transforms(
      source_pos_exp,
      source_quat_exp,
      target_pos_w.reshape(-1, 3),
      target_quat_w.reshape(-1, 4),
    )
    target_pos_source = target_pos_source.view(B, F, 3)
    target_quat_source = target_quat_source.view(B, F, 4)

    return FrameTransformerData(
      target_frame_names=list(self._target_frame_names),
      target_pos_source=target_pos_source,
      target_quat_source=target_quat_source,
      target_pos_w=target_pos_w,
      target_quat_w=target_quat_w,
      source_pos_w=source_pos,
      source_quat_w=source_quat,
    )

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    if not self.cfg.debug_vis:
      return

    data = self.data
    env_indices = list(visualizer.get_env_indices(data.source_pos_w.shape[0]))
    if not env_indices:
      return

    source_pos = data.source_pos_w[env_indices].cpu().numpy()
    source_mat = matrix_from_quat(data.source_quat_w[env_indices]).cpu().numpy()
    target_pos = data.target_pos_w[env_indices].cpu().numpy()
    target_mat = matrix_from_quat(data.target_quat_w[env_indices]).cpu().numpy()

    meansize = visualizer.meansize
    frame_scale = self.cfg.debug_vis_frame_scale * meansize
    axis_radius = self.cfg.debug_vis_axis_radius * meansize
    line_radius = self.cfg.debug_vis_line_radius * meansize
    line_color = self.cfg.debug_vis_line_color
    name = self.cfg.name
    frame_names = self._target_frame_names

    for k, env_idx in enumerate(env_indices):
      visualizer.add_frame(
        position=source_pos[k],
        rotation_matrix=source_mat[k],
        scale=frame_scale,
        axis_radius=axis_radius,
        label=f"{name}_source_env{env_idx}",
      )
      for f, frame_name in enumerate(frame_names):
        visualizer.add_frame(
          position=target_pos[k, f],
          rotation_matrix=target_mat[k, f],
          scale=frame_scale,
          axis_radius=axis_radius,
          label=f"{name}_{frame_name}_env{env_idx}",
        )
        visualizer.add_cylinder(
          start=source_pos[k],
          end=target_pos[k, f],
          radius=line_radius,
          color=line_color,
          label=f"{name}_{frame_name}_line_env{env_idx}",
        )

  # Internal helpers.

  def _resolve_objname(self, body_name: str) -> str:
    if self.cfg.entity:
      return f"{self.cfg.entity}/{body_name}"
    return body_name

  def _add_frame_sensors(
    self,
    scene_spec: mujoco.MjSpec,
    objtype: FrameObjType,
    objname: str,
    existing_sensor_names: set[str],
    slot_idx: int,
  ) -> None:
    """Add ``framepos`` and ``framequat`` sensors for one tracked object."""
    base_name = self.cfg.prefixed_name
    # Encode the slot index into the sensor name to keep names unique even
    # when two FrameTransformer instances target the same body.
    pos_name = f"{base_name}__framepos_{slot_idx}"
    quat_name = f"{base_name}__framequat_{slot_idx}"

    for nm in (pos_name, quat_name):
      if nm in existing_sensor_names:
        raise ValueError(
          f"FrameTransformer '{self.cfg.name}': internal sensor name "
          f"'{nm}' conflicts with an existing sensor in the scene. "
          f"Rename the FrameTransformer to avoid the collision."
        )

    mj_objtype = _OBJECT_TYPE_MAP[objtype]
    kwargs: dict[str, Any] = {
      "objtype": mj_objtype,
      "objname": objname,
    }
    scene_spec.add_sensor(name=pos_name, type=_SENSOR_TYPE_MAP["framepos"], **kwargs)
    scene_spec.add_sensor(name=quat_name, type=_SENSOR_TYPE_MAP["framequat"], **kwargs)
    existing_sensor_names.add(pos_name)
    existing_sensor_names.add(quat_name)

    self._sensor_names.append((pos_name, quat_name))
