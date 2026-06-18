"""Python wrapper for xcamera C library via ctypes.

Loads libxcamere_c_extern.so and exposes the camera model interface
for loading JSON calibration files and projecting 3D world points to image.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ── Library Loading ──────────────────────────────────────────────────────────


def _get_xutils_lib_dir() -> Path:
    """Resolve the directory containing xcamera shared libraries.

    Resolution order (first match wins):
        1. ``XUTILS_LIB_PATH`` environment variable (directory or .so file).
        2. Default legacy path for local development.

    Returns:
        Absolute path to the library directory.
    """
    env_path = os.environ.get("XUTILS_LIB_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_dir():
            return p
        elif p.is_file():
            return p.parent
    return Path("/home/nio/EOL_cpp/xutils/xutils/build/Linux-x86_64/lib")


def _get_so_name() -> str:
    """Return the C-extern shared-library filename (with or without '_c')."""
    lib_dir = _get_xutils_lib_dir()
    candidates = ("libxcamere_c_extern.so", "libxcamera_c_extern.so")
    for name in candidates:
        if (lib_dir / name).exists():
            return name
    return candidates[0]  # fallback


_LIB_DIR = _get_xutils_lib_dir()
_LIB_PATH = _LIB_DIR / _get_so_name()


# ── Function Signatures ──────────────────────────────────────────────────────


def _load_library() -> ctypes.CDLL:
    """Load the xcamera C extern library with proper dependency resolution."""
    # Ensure the library can find its dependencies (libxcamera.so, libxlog.so)
    os.environ.setdefault("LD_LIBRARY_PATH", str(_LIB_DIR))

    # Try loading with RTLD_GLOBAL so dependent .so symbols resolve
    lib = ctypes.CDLL(str(_LIB_PATH), mode=ctypes.RTLD_GLOBAL)
    return lib


_LIB = _load_library()

# void* createICameraModelFromCameraJson(const char* _camera_json_file_name)
_LIB.createICameraModelFromCameraJson.argtypes = [ctypes.c_char_p]
_LIB.createICameraModelFromCameraJson.restype = ctypes.c_void_p

# void* createICameraModelFromJson(const char* _camera_json_file_name, const char* _camera_name)
_LIB.createICameraModelFromJson.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_LIB.createICameraModelFromJson.restype = ctypes.c_void_p

# void deleteInstance(void* ptr)
_LIB.deleteInstance.argtypes = [ctypes.c_void_p]
_LIB.deleteInstance.restype = None

# void worldToImage(void* ptr, const int num_point, double* p3d, double* p2d)
_LIB.worldToImage.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_LIB.worldToImage.restype = None

# void imageToUnitRay(void* ptr, double* p2d, double* p3d)
_LIB.imageToUnitRay.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_LIB.imageToUnitRay.restype = None


# ── Camera Model Class ───────────────────────────────────────────────────────


class CameraModel:
    """Wrapper around xcamera iXCamera C++ model loaded via ctypes."""

    def __init__(self, json_path: str | Path):
        """Load a camera model from a JSON calibration file.

        Args:
            json_path: Path to the camera calibration JSON file.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Camera calibration not found: {json_path}")

        c_path = str(json_path).encode("utf-8")
        self._ptr = _LIB.createICameraModelFromCameraJson(c_path)
        if not self._ptr:
            raise RuntimeError(f"Failed to load camera model from {json_path}")
        self._json_path = json_path

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _LIB.deleteInstance(self._ptr)
            self._ptr = None

    # ── Projections ──────────────────────────────────────────────────────────

    def world_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D world points to 2D image coordinates.

        Args:
            points_3d: Array of shape (N, 3) with 3D points in camera coordinate
                system (or the coordinate system the extrinsics were defined for).

        Returns:
            Array of shape (N, 2) with pixel coordinates (u, v).
        """
        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            raise ValueError(f"points_3d must be (N, 3), got {points_3d.shape}")

        n = points_3d.shape[0]
        p3d = np.ascontiguousarray(points_3d.flatten(), dtype=np.float64)
        p2d = np.empty(n * 2, dtype=np.float64)

        _LIB.worldToImage(
            self._ptr,
            n,
            p3d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            p2d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return p2d.reshape(n, 2)

    def image_to_unit_ray(self, points_2d: np.ndarray) -> np.ndarray:
        """Back-project 2D image points to unit rays in 3D camera space.

        Args:
            points_2d: Array of shape (N, 2) with pixel coordinates.

        Returns:
            Array of shape (N, 3) with unit direction vectors.
        """
        if points_2d.ndim != 2 or points_2d.shape[1] != 2:
            raise ValueError(f"points_2d must be (N, 2), got {points_2d.shape}")

        n = points_2d.shape[0]
        p2d = np.ascontiguousarray(points_2d.flatten(), dtype=np.float64)
        p3d = np.empty(n * 3, dtype=np.float64)

        for i in range(n):
            _LIB.imageToUnitRay(
                self._ptr,
                (p2d.ctypes.data + i * 2 * 8).as_void(),  # type: ignore[attr-defined]
                (p3d.ctypes.data + i * 3 * 8).as_void(),  # type: ignore[attr-defined]
            )

        return p3d.reshape(n, 3)

    def image_to_unit_ray_single(self, u: float, v: float) -> np.ndarray:
        """Back-project a single 2D point to a unit ray."""
        p2d = np.array([u, v], dtype=np.float64)
        p3d = np.empty(3, dtype=np.float64)

        _LIB.imageToUnitRay(
            self._ptr,
            p2d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            p3d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return p3d

    def get_image_size(self) -> tuple[int, int]:
        """Return (width, height) from the calibration JSON file.

        Returns:
            (width, height) in pixels as stored in intrinsic_param.
        """
        import json

        with open(self._json_path) as f:
            data = json.load(f)
        intrinsic = data.get("intrinsic_param", data)
        w = int(intrinsic.get("camera_width", 1920))
        h = int(intrinsic.get("camera_height", 1080))
        return w, h

    def get_name(self) -> str:
        """Return camera name from calibration JSON."""
        import json

        with open(self._json_path) as f:
            data = json.load(f)
        return data.get("name", "unknown")


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_camera(json_path: str | Path) -> CameraModel:
    """Convenience factory: load a CameraModel from JSON path."""
    return CameraModel(json_path)


def batch_world_to_image(
    cameras: List[CameraModel], points_3d_batch: List[np.ndarray]
) -> List[np.ndarray]:
    """Project 3D points for multiple cameras in one call list.

    Args:
        cameras: List of CameraModel instances.
        points_3d_batch: List of (N, 3) arrays, one per camera.

    Returns:
        List of (N, 2) image-coordinate arrays.
    """
    return [cam.world_to_image(pts) for cam, pts in zip(cameras, points_3d_batch)]
