"""Tests for naviguard.fusion.geometry"""

import math

import pytest

from naviguard.fusion.geometry import (
    bbox_height_to_range,
    bearing_to_absolute,
    enu_to_latlon,
    heading_to_velocity,
    knots_to_ms,
    latlon_to_enu,
    pixel_to_bearing,
    polar_to_enu,
)


class TestPixelToBearing:
    def test_centre_pixel_is_zero(self):
        bearing = pixel_to_bearing(640, img_width=1280, cam_fov_deg=90)
        assert bearing == pytest.approx(0.0, abs=1e-9)

    def test_right_edge_is_positive(self):
        bearing = pixel_to_bearing(1280, img_width=1280, cam_fov_deg=90)
        assert bearing == pytest.approx(math.radians(45.0), rel=1e-6)

    def test_left_edge_is_negative(self):
        bearing = pixel_to_bearing(0, img_width=1280, cam_fov_deg=90)
        assert bearing == pytest.approx(-math.radians(45.0), rel=1e-6)

    def test_narrow_fov(self):
        # Right edge of a 640-wide image with 30° FoV → +15°
        bearing = pixel_to_bearing(640, img_width=640, cam_fov_deg=30)
        assert bearing == pytest.approx(math.radians(15.0), rel=1e-6)


class TestBearingToAbsolute:
    def test_zero_relative(self):
        abs_b = bearing_to_absolute(0.0, heading_deg=90.0)
        assert abs_b == pytest.approx(90.0, abs=1e-6)

    def test_wrap_around(self):
        abs_b = bearing_to_absolute(math.radians(20.0), heading_deg=350.0)
        assert abs_b == pytest.approx(10.0, abs=1e-3)


class TestPolarToEnu:
    def test_due_north(self):
        east, north = polar_to_enu(100.0, bearing_rad=0.0)
        assert east == pytest.approx(0.0, abs=1e-6)
        assert north == pytest.approx(100.0, rel=1e-6)

    def test_due_east(self):
        east, north = polar_to_enu(100.0, bearing_rad=math.pi / 2)
        assert east == pytest.approx(100.0, rel=1e-6)
        assert north == pytest.approx(0.0, abs=1e-6)


class TestLatLonEnu:
    def test_round_trip(self):
        origin_lat, origin_lon = 51.5, -0.09
        lat, lon = 51.501, -0.088
        east, north = latlon_to_enu(lat, lon, origin_lat, origin_lon)
        lat2, lon2 = enu_to_latlon(east, north, origin_lat, origin_lon)
        assert lat2 == pytest.approx(lat, rel=1e-5)
        assert lon2 == pytest.approx(lon, rel=1e-5)

    def test_north_positive(self):
        _, north = latlon_to_enu(51.51, -0.09, 51.5, -0.09)
        assert north > 0

    def test_east_positive(self):
        east, _ = latlon_to_enu(51.5, -0.08, 51.5, -0.09)
        assert east > 0


class TestKnotsToMs:
    def test_one_knot(self):
        assert knots_to_ms(1.0) == pytest.approx(0.514444, rel=1e-4)

    def test_zero(self):
        assert knots_to_ms(0.0) == pytest.approx(0.0)


class TestHeadingToVelocity:
    def test_north(self):
        vx, vy = heading_to_velocity(0.0, 1.0)
        assert vx == pytest.approx(0.0, abs=1e-9)
        assert vy == pytest.approx(1.0, rel=1e-6)

    def test_east(self):
        vx, vy = heading_to_velocity(90.0, 1.0)
        assert vx == pytest.approx(1.0, rel=1e-6)
        assert vy == pytest.approx(0.0, abs=1e-6)

    def test_south(self):
        vx, vy = heading_to_velocity(180.0, 1.0)
        assert vx == pytest.approx(0.0, abs=1e-6)
        assert vy == pytest.approx(-1.0, rel=1e-6)


class TestBboxHeightToRange:
    def test_large_box_is_close(self):
        r = bbox_height_to_range(200, img_height=720, camera_height_m=3.0)
        assert r < 100  # large bbox → very close

    def test_tiny_box_is_far(self):
        r = bbox_height_to_range(5, img_height=720, camera_height_m=3.0)
        assert r > 200  # small bbox → far away

    def test_zero_height_returns_max(self):
        r = bbox_height_to_range(0)
        assert r == pytest.approx(1500.0)
