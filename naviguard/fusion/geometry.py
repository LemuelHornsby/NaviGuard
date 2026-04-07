"""Geometry utilities for NaviGuard.

Provides helpers for:
- Converting pixel x-coordinate → bearing relative to ship heading
- Estimating target range from bounding-box height (heuristic)
- Converting polar (range, bearing) → ENU (East-North-Up) coordinates
- Converting lat/lon to local ENU frame
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Pixel → bearing
# ---------------------------------------------------------------------------

def pixel_to_bearing(
    px: float,
    img_width: int,
    cam_fov_deg: float,
) -> float:
    """Convert a pixel x-coordinate to a relative bearing in radians.

    Parameters
    ----------
    px : float
        Pixel x-coordinate (0 = left edge of image).
    img_width : int
        Total image width in pixels.
    cam_fov_deg : float
        Horizontal field of view of the camera in degrees.

    Returns
    -------
    float
        Relative bearing in radians. Positive = to starboard (right),
        negative = to port (left).
    """
    rel = (px - img_width / 2.0) / (img_width / 2.0)   # in [-1, 1]
    angle_rad = rel * math.radians(cam_fov_deg / 2.0)
    return angle_rad


def bearing_to_absolute(relative_bearing_rad: float, heading_deg: float) -> float:
    """Convert a camera-relative bearing to an absolute compass bearing.

    Parameters
    ----------
    relative_bearing_rad : float
        Bearing relative to the ship's bow in radians.
    heading_deg : float
        Own ship heading in degrees (0 = North, 90 = East, …).

    Returns
    -------
    float
        Absolute bearing in degrees [0, 360).
    """
    abs_deg = heading_deg + math.degrees(relative_bearing_rad)
    return abs_deg % 360.0


# ---------------------------------------------------------------------------
# Range estimation from bounding-box geometry
# ---------------------------------------------------------------------------

# Typical apparent heights (px at 720 p, 90° FOV) for each range bin.
# These constants are rough heuristics; calibrate from NaviSense ground truth.
_RANGE_BINS = [
    (150, 30.0),    # bbox_h > 150 px → ≈ 30 m
    (80, 80.0),
    (40, 200.0),
    (20, 400.0),
    (10, 800.0),
    (0, 1500.0),    # very small → far away
]


def bbox_height_to_range(
    bbox_height_px: float,
    img_height: int = 720,
    camera_height_m: float = 3.0,
) -> float:
    """Heuristic range estimate from bounding-box height.

    Uses a simple geometric model:  range ≈ (camera_height × img_height)
    / (bbox_height × tan(vfov/2)).  Falls back to bin lookup when
    camera_height is 0.

    Parameters
    ----------
    bbox_height_px : float
        Height of the bounding box in pixels.
    img_height : int
        Total image height in pixels.
    camera_height_m : float
        Camera mount height above the water line in metres.

    Returns
    -------
    float
        Estimated range in metres (>= 1.0).
    """
    if bbox_height_px <= 0:
        return 1500.0

    # Assume vertical FoV ≈ 56° (for a 90° HFOV 16:9 camera)
    v_fov_rad = math.radians(56.0)
    angle_per_px = v_fov_rad / img_height  # radians per pixel

    # Angular height of the object
    theta = bbox_height_px * angle_per_px

    if camera_height_m > 0 and theta > 1e-6:
        range_m = camera_height_m / math.tan(theta)
        return max(1.0, range_m)

    # Bin fallback
    for threshold, r in _RANGE_BINS:
        if bbox_height_px > threshold:
            return r
    return 1500.0


# ---------------------------------------------------------------------------
# Polar → ENU
# ---------------------------------------------------------------------------

def polar_to_enu(range_m: float, bearing_rad: float) -> Tuple[float, float]:
    """Convert polar (range, relative bearing) to ENU (East, North).

    The bearing is relative to the ship's bow.  Positive = starboard.

    Returns
    -------
    (east_m, north_m) : Tuple[float, float]
        Approximate displacement from own ship in metres.
    """
    east = range_m * math.sin(bearing_rad)
    north = range_m * math.cos(bearing_rad)
    return east, north


# ---------------------------------------------------------------------------
# Lat/lon ↔ ENU  (flat-Earth approximation, valid for short distances ≤ ~50 km)
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000.0


def latlon_to_enu(
    lat: float,
    lon: float,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[float, float]:
    """Convert geodetic lat/lon to local ENU (East, North) in metres.

    Parameters
    ----------
    lat, lon : float
        Target position in decimal degrees.
    origin_lat, origin_lon : float
        Reference origin (own ship position) in decimal degrees.

    Returns
    -------
    (east_m, north_m)
    """
    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)
    north = dlat * _EARTH_RADIUS_M
    east = dlon * _EARTH_RADIUS_M * math.cos(math.radians(origin_lat))
    return east, north


def enu_to_latlon(
    east_m: float,
    north_m: float,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[float, float]:
    """Convert local ENU offsets back to geodetic lat/lon."""
    dlat = north_m / _EARTH_RADIUS_M
    dlon = east_m / (_EARTH_RADIUS_M * math.cos(math.radians(origin_lat)))
    return origin_lat + math.degrees(dlat), origin_lon + math.degrees(dlon)


# ---------------------------------------------------------------------------
# Speed conversion
# ---------------------------------------------------------------------------

def knots_to_ms(knots: float) -> float:
    """Convert knots to metres per second."""
    return knots * 0.514444


def heading_to_velocity(heading_deg: float, speed_ms: float) -> Tuple[float, float]:
    """Convert compass heading + speed into (vx, vy) ENU velocity vector.

    Returns
    -------
    (vx, vy) in m/s where vx = east component, vy = north component.
    """
    heading_rad = math.radians(heading_deg)
    vx = speed_ms * math.sin(heading_rad)   # east
    vy = speed_ms * math.cos(heading_rad)   # north
    return vx, vy
