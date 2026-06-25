"""Front120 BEV (Bird's Eye View) projector.

Projects a single Front120 camera image onto a virtual pinhole BEV camera
positioned above and in front of the vehicle, looking downward.

Reference: /home/nio/EOL_cpp/power_swap_auto_park/calibrators/psap_utils/psap_cfs.cpp
    projectToGround() implementation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .camera_wrapper import CameraModel


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class Front120BevConfig:
    """Configuration for Front120 BEV projection.

    Defaults produce a 1280x720 BEV image with the virtual camera at
    (5, 0, 10) in base/ground coordinates, pitched 15° forward looking downward.
    """

    bev_width: int = 1280
    bev_height: int = 720
    bev_fx: float = 300.0
    bev_fy: float = 300.0
    # BEV camera position in base/ground coordinates (meters)
    bev_pos_x: float = 5.0  # forward
    bev_pos_y: float = 0.0  # left/right
    bev_pos_z: float = 10.0  # up
    # Pitch angle: forward tilt from vertical-down (degrees)
    # 0 = straight down, positive = tilting forward (toward +X)
    bev_pitch_deg: float = 15.0


# ── Front120 BEV Projector ───────────────────────────────────────────────────


class Front120BevProjector:
    """Projects Front120 fisheye image to a virtual overhead BEV camera."""

    def __init__(self, config: Front120BevConfig | None = None):
        self.cfg = config or Front120BevConfig()
        self._camera: CameraModel | None = None
        self._remap_x: np.ndarray | None = None
        self._remap_y: np.ndarray | None = None

        # Resolution scaling: video vs calibration
        self._calib_width: int = 0
        self._calib_height: int = 0
        self._scale_u: float = 1.0
        self._scale_v: float = 1.0

        # Front120 extrinsic: Base -> Camera (Tb2c)
        self._T_b2c: np.ndarray | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def init(
        self,
        camera_json: str | Path,
        video_width: int,
        video_height: int,
    ) -> "Front120BevProjector":
        """Initialize projector with calibration and video dimensions.

        Args:
            camera_json: Path to Front120 (front_wide) calibration JSON.
            video_width: Actual video frame width in pixels.
            video_height: Actual video frame height in pixels.
        """
        self._camera = CameraModel(camera_json)
        self._calib_width, self._calib_height = self._camera.get_image_size()

        # Compute resolution scaling factor
        self._scale_u = video_width / self._calib_width
        self._scale_v = video_height / self._calib_height

        # Load Tb2c from JSON (same logic as bev_projector.py)
        self._T_b2c = self._load_tb2c(camera_json)

        # Pre-compute remap table
        self._compute_remap_table()

        return self

    def get_remap_tables(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (remap_x, remap_y) for cv2.remap."""
        if self._remap_x is None or self._remap_y is None:
            raise RuntimeError("Projector not initialized. Call init() first.")
        return self._remap_x, self._remap_y

    def get_bev_size(self) -> Tuple[int, int]:
        """Return (width, height) of BEV output."""
        return self.cfg.bev_width, self.cfg.bev_height

    # ── Internal: Load Tb2c from JSON ────────────────────────────────────────

    @staticmethod
    def _load_tb2c(json_path: str | Path) -> np.ndarray:
        """Build Tb2c (Base -> Camera) transform from calibration JSON.

        JSON stores Tc2b (Camera -> Base) as Rodrigues rotation + translation.
        We invert to get Tb2c.
        """
        with open(json_path) as f:
            data = json.load(f)
        extrinsic = data["extrinsic_param"]
        rotation = np.array(extrinsic["rotation"], dtype=np.float64)
        translation = np.array(extrinsic["translation"], dtype=np.float64)

        # Build Tc2b
        rvec = rotation.reshape(3, 1)
        R_c2b, _ = cv2.Rodrigues(rvec)
        T_c2b = np.eye(4, dtype=np.float64)
        T_c2b[:3, :3] = R_c2b
        T_c2b[:3, 3] = translation

        # Invert to get Tb2c
        T_b2c = np.linalg.inv(T_c2b)
        return T_b2c

    # ── Internal: Compute Remap Table ────────────────────────────────────────

    def _compute_remap_table(self) -> None:
        """Pre-compute remap_x / remap_y for all BEV pixels.

        Algorithm (from C++ projectToGround):
        1. For each BEV pixel (u, v):
           uray_bev = pinhole_image_to_unit_ray(u, v)  [BEV camera coords]
        2. uray_g = R_bev2ground @ uray_bev            [Ground coords]
        3. ground_pt = C_bev + len * uray_g
           where len = -C_bev.z / uray_g.z
        4. cam_pt = (Tb2c @ [ground_pt, 1])[:3]        [Front120 camera coords]
        5. img_uv = camera.world_to_image(cam_pt)       [Image pixels]
        6. Apply resolution scaling: img_uv *= scale
        """
        w = self.cfg.bev_width
        h = self.cfg.bev_height
        rx = np.full((h, w), -1.0, dtype=np.float32)
        ry = np.full((h, w), -1.0, dtype=np.float32)

        # BEV virtual camera intrinsics
        fx = self.cfg.bev_fx
        fy = self.cfg.bev_fy
        cx = w / 2.0
        cy = h / 2.0

        # BEV camera position in Ground coords
        C_bev = np.array(
            [
                self.cfg.bev_pos_x,
                self.cfg.bev_pos_y,
                self.cfg.bev_pos_z,
            ],
            dtype=np.float64,
        )

        # BEV camera rotation matrix: maps vectors from BEV camera frame to Ground frame.
        #
        # Base orientation (pitch=0): image up=forward (+X), image right=right (-Y),
        #                              optical axis=down (-Z).
        #
        # With forward pitch: rotate around BEV X-axis by +pitch_deg.
        # Positive pitch = optical axis tilts toward +X (forward).
        pitch_rad = math.radians(self.cfg.bev_pitch_deg)
        cp = math.cos(pitch_rad)
        sp = math.sin(pitch_rad)

        # R_bev2ground = R_base @ Rx(pitch)
        # where R_base maps BEV axes to Ground at pitch=0,
        # and Rx rotates around BEV X-axis.
        #
        # Columns of R_bev2ground are BEV basis vectors expressed in Ground frame:
        #   col 0 (BEV X / image right) -> Ground -Y
        #   col 1 (BEV Y / image down)  -> Ground -X with pitch tilt
        #   col 2 (BEV Z / optical axis)-> Ground -Z with pitch tilt
        R_bev2ground = np.array(
            [
                [0.0, -cp, sp],
                [-1.0, 0.0, 0.0],
                [0.0, -sp, -cp],
            ],
            dtype=np.float64,
        )

        # Build grid of all BEV pixels
        uu, vv = np.meshgrid(
            np.arange(w, dtype=np.float64),
            np.arange(h, dtype=np.float64),
        )
        n = w * h
        u_flat = uu.ravel()
        v_flat = vv.ravel()

        # Step 1: BEV pixel -> unit ray in BEV camera coords
        x_cam = (u_flat - cx) / fx
        y_cam = (v_flat - cy) / fy
        z_cam = np.ones(n, dtype=np.float64)

        norms = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
        rays_bev = np.stack([x_cam / norms, y_cam / norms, z_cam / norms], axis=1)

        # Step 2: BEV camera ray -> Ground coords
        rays_g = (R_bev2ground @ rays_bev.T).T

        # Step 3: Intersect with ground plane Z=0
        # ground_pt = C_bev + len * uray_g
        # ground_pt.z = 0  =>  C_bev.z + len * uray_g.z = 0
        # len = -C_bev.z / uray_g.z
        uray_gz = rays_g[:, 2]
        valid_mask = uray_gz < -1e-6  # ray must point downward

        len_arr = np.full(n, np.nan, dtype=np.float64)
        len_arr[valid_mask] = -C_bev[2] / uray_gz[valid_mask]

        # Only keep positive-length intersections (in front of camera)
        pos_mask = valid_mask & (len_arr > 0)

        ground_pts = np.full((n, 3), np.nan, dtype=np.float64)
        ground_pts[pos_mask] = C_bev + len_arr[pos_mask][:, None] * rays_g[pos_mask]

        # Step 4: Ground point -> Front120 camera coords
        P_g = np.hstack(
            [
                ground_pts,
                np.ones((n, 1), dtype=np.float64),
            ]
        )
        P_c = (self._T_b2c @ P_g.T).T[:, :3]

        # Step 5: Project to image via xcamera
        valid_proj = pos_mask & np.isfinite(P_c).all(axis=1)
        img_pts = np.full((n, 2), np.nan, dtype=np.float64)
        if valid_proj.any():
            img_pts[valid_proj] = self._camera.world_to_image(P_c[valid_proj])

        # Step 6: Apply resolution scaling
        img_pts[:, 0] *= self._scale_u
        img_pts[:, 1] *= self._scale_v

        # Store remap tables
        rx_flat = img_pts[:, 0].astype(np.float32)
        ry_flat = img_pts[:, 1].astype(np.float32)

        # Mark invalid pixels
        rx_flat[~valid_proj] = -1.0
        ry_flat[~valid_proj] = -1.0

        self._remap_x = rx_flat.reshape(h, w)
        self._remap_y = ry_flat.reshape(h, w)

    def get_valid_mask(self) -> np.ndarray:
        """Return boolean mask of valid BEV pixels."""
        if self._remap_x is None:
            raise RuntimeError("Projector not initialized")
        return (self._remap_x >= 0) & (self._remap_y >= 0)
