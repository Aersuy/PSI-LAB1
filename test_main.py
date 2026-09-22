"""Unit tests for the 3D solar system simulator.

Run with:  python -m unittest -v
Only the standard library is needed beyond what main.py already requires.
"""

import contextlib
import io
import math
import time
import unittest

import matplotlib
matplotlib.use("Agg")          # headless: must precede main's pyplot import

import numpy as np

import main
from main import (PLANETS, PLUTO, SolarSystem, date_to_jd, format_jd,
                  jd_to_date, marker_size, orbital_to_ecliptic, parse_args,
                  solve_kepler, today_jd, warp)

ALL_BODIES = [*PLANETS, PLUTO]


class FakeEvent:
    """Stand-in for a matplotlib key or scroll event."""

    def __init__(self, key=None, button=None, inaxes=None):
        self.key = key
        self.button = button
        self.inaxes = inaxes


# --------------------------------------------------------------------------- #
# Orbital mechanics
# --------------------------------------------------------------------------- #

class TestKeplerSolver(unittest.TestCase):
    """M = E - e*sin(E) must hold to near machine precision."""

    def test_residual_is_negligible(self):
        mean_anom = np.linspace(-np.pi, np.pi, 2001)
        for ecc in (0.0, 0.0167, 0.2056, 0.2488, 0.6, 0.9):
            with self.subTest(e=ecc):
                ecc_anom = solve_kepler(mean_anom, ecc)
                residual = ecc_anom - ecc * np.sin(ecc_anom) - mean_anom
                self.assertLess(np.abs(residual).max(), 1e-9)

    def test_circular_orbit_is_identity(self):
        mean_anom = np.linspace(-np.pi, np.pi, 101)
        np.testing.assert_allclose(solve_kepler(mean_anom, 0.0), mean_anom)

    def test_accepts_scalar(self):
        ecc_anom = float(solve_kepler(1.0, 0.2))
        self.assertAlmostEqual(ecc_anom - 0.2 * math.sin(ecc_anom), 1.0, places=10)

    def test_preserves_shape(self):
        self.assertEqual(solve_kepler(np.zeros((4, 7)), 0.3).shape, (4, 7))

    def test_accepts_per_sample_eccentricity(self):
        # position() passes a time-varying e array when jd is an array.
        mean_anom = np.linspace(0.0, 2.0 * np.pi, 50)
        ecc = np.linspace(0.01, 0.4, 50)
        ecc_anom = solve_kepler(mean_anom, ecc)
        residual = ecc_anom - ecc * np.sin(ecc_anom) - mean_anom
        self.assertLess(np.abs(residual).max(), 1e-9)

    def test_extreme_eccentricity_still_converges(self):
        mean_anom = np.linspace(-np.pi, np.pi, 501)
        ecc_anom = solve_kepler(mean_anom, 0.95)
        self.assertTrue(np.all(np.isfinite(ecc_anom)))
        residual = ecc_anom - 0.95 * np.sin(ecc_anom) - mean_anom
        self.assertLess(np.abs(residual).max(), 1e-8)


class TestOrbitalRotation(unittest.TestCase):

    def test_zero_angles_is_identity(self):
        out = orbital_to_ecliptic(np.array([2.0]), np.array([3.0]), 0.0, 0.0, 0.0)
        np.testing.assert_allclose(out, [[2.0, 3.0, 0.0]], atol=1e-12)

    def test_uninclined_orbit_stays_in_the_plane(self):
        x, y = np.array([1.0, -2.0]), np.array([0.5, 4.0])
        out = orbital_to_ecliptic(x, y, 0.7, 1.2, 0.0)
        np.testing.assert_allclose(out[:, 2], 0.0, atol=1e-12)

    def test_rotation_preserves_length(self):
        x, y = np.array([1.0, -2.0, 0.3]), np.array([0.5, 4.0, -1.1])
        out = orbital_to_ecliptic(x, y, 0.7, 1.2, 0.4)
        np.testing.assert_allclose(np.linalg.norm(out, axis=-1),
                                   np.hypot(x, y), rtol=1e-12)

    def test_inclination_lifts_out_of_plane(self):
        out = orbital_to_ecliptic(np.array([1.0]), np.array([0.0]),
                                  arg_peri=np.pi / 2, node=0.0, inc=0.5)
        self.assertAlmostEqual(float(out[0, 2]), math.sin(0.5), places=12)

    def test_height_follows_the_standard_identity(self):
        # z = r * sin(i) * sin(arg_peri + in-plane angle)
        radius, arg_peri, inc = 2.5, 0.8, 0.3
        theta = np.linspace(0.0, 2.0 * np.pi, 37)
        out = orbital_to_ecliptic(radius * np.cos(theta), radius * np.sin(theta),
                                  arg_peri, 1.1, inc)
        np.testing.assert_allclose(
            out[:, 2], radius * math.sin(inc) * np.sin(arg_peri + theta),
            atol=1e-12)

    def test_node_does_not_affect_height(self):
        # The ascending node only spins the orbit about the z axis.
        x, y = np.array([1.0, -2.0]), np.array([0.5, 4.0])
        first = orbital_to_ecliptic(x, y, 0.7, 0.0, 0.4)
        second = orbital_to_ecliptic(x, y, 0.7, 2.3, 0.4)
        np.testing.assert_allclose(first[:, 2], second[:, 2], atol=1e-12)

    def test_output_shape(self):
        out = orbital_to_ecliptic(np.zeros(17), np.zeros(17), 0.1, 0.2, 0.3)
        self.assertEqual(out.shape, (17, 3))


class TestBody(unittest.TestCase):

    def test_period_matches_keplers_third_law(self):
        expected = {"Mercury": 87.97, "Earth": 365.26, "Mars": 686.98,
                    "Jupiter": 4332.6, "Neptune": 60189.0}
        for body in ALL_BODIES:
            if body.name in expected:
                with self.subTest(body=body.name):
                    self.assertAlmostEqual(body.period_days, expected[body.name],
                                           delta=0.01 * expected[body.name])

    def test_radius_stays_between_perihelion_and_aphelion(self):
        jds = np.linspace(date_to_jd(1900, 1, 1), date_to_jd(2050, 1, 1), 2000)
        for body in ALL_BODIES:
            with self.subTest(body=body.name):
                radius = np.linalg.norm(body.position(jds), axis=-1)
                self.assertGreaterEqual(radius.min(), body.a * (1 - body.e) - 0.02)
                self.assertLessEqual(radius.max(), body.a * (1 + body.e) + 0.02)

    def test_eccentric_orbit_actually_varies(self):
        jds = np.linspace(main.J2000, main.J2000 + 88.0, 400)
        radius = np.linalg.norm(PLANETS[0].position(jds), axis=-1)   # Mercury
        self.assertGreater(radius.max() - radius.min(), 0.1)

    def test_orbit_closes_after_one_period(self):
        for body in ALL_BODIES:
            with self.subTest(body=body.name):
                drift = np.linalg.norm(body.position(main.J2000 + body.period_days)
                                       - body.position(main.J2000))
                self.assertLess(drift, 0.08 * body.a)

    def test_earth_longitude_at_j2000(self):
        # Earth's heliocentric longitude at J2000.0 is ~100.46 deg: the Sun's
        # geocentric longitude of ~280.46 deg, turned around.
        earth = next(b for b in PLANETS if b.name == "Earth")
        pos = earth.position(main.J2000)
        lon = math.degrees(math.atan2(pos[1], pos[0])) % 360.0
        self.assertAlmostEqual(lon, 100.46, delta=0.5)
        self.assertAlmostEqual(float(np.linalg.norm(pos)), 0.983, delta=0.01)

    def test_scalar_and_array_positions_agree(self):
        earth = next(b for b in PLANETS if b.name == "Earth")
        jds = np.array([main.J2000, main.J2000 + 100.0])
        np.testing.assert_allclose(earth.position(jds)[1],
                                   earth.position(main.J2000 + 100.0), rtol=1e-12)

    def test_position_shapes(self):
        earth = next(b for b in PLANETS if b.name == "Earth")
        self.assertEqual(earth.position(main.J2000).shape, (3,))
        self.assertEqual(earth.position(np.full(5, main.J2000)).shape, (5, 3))

    def test_orbit_path_is_a_closed_loop(self):
        for body in ALL_BODIES:
            with self.subTest(body=body.name):
                path = body.orbit_path(main.J2000)
                self.assertEqual(path.shape, (main.ORBIT_SAMPLES, 3))
                np.testing.assert_allclose(path[0], path[-1], atol=1e-9)

    def test_body_sits_on_its_own_drawn_orbit(self):
        for body in ALL_BODIES:
            with self.subTest(body=body.name):
                path = body.orbit_path(main.J2000)
                gap = np.linalg.norm(path - body.position(main.J2000), axis=-1).min()
                self.assertLess(gap, 0.02 * body.a)

    def test_orbits_are_not_coplanar(self):
        jds = np.linspace(date_to_jd(1900, 1, 1), date_to_jd(2050, 1, 1), 500)
        height = {b.name: np.abs(b.position(jds)[:, 2]).max() for b in ALL_BODIES}
        self.assertGreater(height["Pluto"], 5.0)        # 17 deg inclination
        self.assertGreater(height["Mercury"], 0.03)     # 7 deg inclination
        self.assertLess(height["Earth"], 0.001)         # Earth defines the ecliptic

    def test_secular_rates_drift_the_elements(self):
        jupiter = next(b for b in PLANETS if b.name == "Jupiter")
        a_then = jupiter.elements_at(main.J2000)[0]
        a_later = jupiter.elements_at(main.J2000 + main.DAYS_PER_CENTURY)[0]
        self.assertAlmostEqual(float(a_later - a_then), jupiter.a_dot, places=10)

    def test_elements_returned_in_radians(self):
        inc = PLUTO.elements_at(main.J2000)[2]
        self.assertAlmostEqual(float(inc), math.radians(17.14), places=4)


class TestElementTable(unittest.TestCase):

    def test_expected_bodies_ordered_outward(self):
        self.assertEqual([b.name for b in PLANETS],
                         ["Mercury", "Venus", "Earth", "Mars",
                          "Jupiter", "Saturn", "Uranus", "Neptune"])
        semi_major = [b.a for b in ALL_BODIES]
        self.assertEqual(semi_major, sorted(semi_major))

    def test_elements_are_physically_sane(self):
        for body in ALL_BODIES:
            with self.subTest(body=body.name):
                self.assertGreater(body.a, 0.0)
                self.assertTrue(0.0 <= body.e < 1.0)      # bound, elliptical
                self.assertTrue(-90.0 < body.inc < 90.0)
                self.assertGreater(body.radius_km, 0.0)

    def test_bodies_are_immutable(self):
        with self.assertRaises(Exception):
            PLANETS[0].a = 99.0


# --------------------------------------------------------------------------- #
# Calendar and display helpers
# --------------------------------------------------------------------------- #

class TestCalendar(unittest.TestCase):

    def test_known_julian_dates(self):
        # J2000.0 is 2000-01-01 12:00 = JD 2451545.0, so midnight is 0.5 less.
        self.assertAlmostEqual(date_to_jd(2000, 1, 1), 2451544.5, places=6)
        self.assertAlmostEqual(date_to_jd(2000, 1, 1.5), main.J2000, places=6)
        self.assertAlmostEqual(date_to_jd(1969, 7, 20), 2440422.5, places=6)

    def test_roundtrip(self):
        for date in [(1800, 1, 1), (1900, 2, 28), (1969, 7, 20), (2000, 1, 1),
                     (2024, 2, 29), (2026, 9, 22), (2050, 12, 31)]:
            with self.subTest(date=date):
                self.assertEqual(jd_to_date(date_to_jd(*date)), date)

    def test_consecutive_days(self):
        base = date_to_jd(2026, 2, 27)
        self.assertEqual(jd_to_date(base + 1), (2026, 2, 28))
        self.assertEqual(jd_to_date(base + 2), (2026, 3, 1))   # 2026: no leap day

    def test_leap_day_exists_in_2024(self):
        self.assertEqual(jd_to_date(date_to_jd(2024, 2, 28) + 1), (2024, 2, 29))

    def test_format(self):
        self.assertEqual(format_jd(date_to_jd(2026, 9, 22)), "2026-09-22")
        self.assertTrue(format_jd(date_to_jd(-100, 1, 1)).endswith("BCE"))

    def test_today_is_plausible(self):
        self.assertAlmostEqual(today_jd(), date_to_jd(*time.gmtime()[:3]), places=6)


class TestWarp(unittest.TestCase):

    def test_real_scale_is_the_identity(self):
        pos = np.array([1.5, -2.5, 0.3])
        np.testing.assert_array_equal(warp(pos, False), pos)

    def test_unit_radius_is_a_fixed_point(self):
        np.testing.assert_allclose(warp(np.array([1.0, 0.0, 0.0]), True),
                                   [1.0, 0.0, 0.0], atol=1e-12)

    def test_is_monotonic_in_radius(self):
        radii = np.array([0.3, 0.39, 1.0, 5.2, 30.1, 49.3])
        pts = np.column_stack([radii, np.zeros_like(radii), np.zeros_like(radii)])
        warped = np.linalg.norm(warp(pts, True), axis=-1)
        self.assertTrue(np.all(np.diff(warped) > 0))

    def test_compresses_the_outer_system(self):
        def span(radius):
            return np.linalg.norm(warp(np.array([radius, 0.0, 0.0]), True))
        self.assertLess(span(30.07) / span(0.39), 30.07 / 0.39)

    def test_preserves_direction(self):
        vec = np.array([3.0, -4.0, 12.0])
        warped = warp(vec, True)
        cos = np.dot(vec, warped) / (np.linalg.norm(vec) * np.linalg.norm(warped))
        self.assertAlmostEqual(float(cos), 1.0, places=12)

    def test_origin_does_not_divide_by_zero(self):
        out = warp(np.zeros(3), True)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_applies_rowwise_to_arrays(self):
        warped = warp(np.array([[1.0, 0.0, 0.0], [0.0, 30.0, 0.0]]), True)
        self.assertEqual(warped.shape, (2, 3))
        np.testing.assert_allclose(warped[0], [1.0, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(float(warped[1, 1]),
                               30.0 ** main.COMPRESS_EXPONENT, places=10)


class TestMarkerSize(unittest.TestCase):

    def test_earth_is_the_reference(self):
        self.assertAlmostEqual(marker_size(main.EARTH_RADIUS_KM, base=5.0), 5.0)

    def test_bigger_body_bigger_marker(self):
        sizes = [marker_size(b.radius_km)
                 for b in sorted(ALL_BODIES, key=lambda b: b.radius_km)]
        self.assertEqual(sizes, sorted(sizes))

    def test_sun_is_compressed_not_literal(self):
        ratio = marker_size(main.SUN_RADIUS_KM) / marker_size(main.EARTH_RADIUS_KM)
        self.assertLess(ratio, 10.0)        # the true radius ratio is ~109
        self.assertGreater(ratio, 2.0)


class TestParseArgs(unittest.TestCase):

    def test_defaults(self):
        args = parse_args([])
        self.assertIsNone(args.date)
        self.assertEqual(args.speed, 20.0)
        self.assertFalse(args.real_scale)
        self.assertFalse(args.no_pluto)

    def test_explicit_values(self):
        args = parse_args(["--date", "2030-01-01", "--speed", "250",
                           "--real-scale", "--no-pluto"])
        self.assertEqual(args.date, "2030-01-01")
        self.assertEqual(args.speed, 250.0)
        self.assertTrue(args.real_scale)
        self.assertTrue(args.no_pluto)

    def test_malformed_date_is_rejected(self):
        with self.assertRaises(SystemExit):
            main.main(["--date", "not-a-date"])

    def test_non_numeric_speed_is_rejected(self):
        # argparse prints its usage to stderr on the way out; swallow it so the
        # test run stays readable.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--speed", "fast"])


# --------------------------------------------------------------------------- #
# The simulation object, driven headlessly through the Agg backend
# --------------------------------------------------------------------------- #

class SimulationTestCase(unittest.TestCase):

    def setUp(self):
        self.sim = SolarSystem(ALL_BODIES, start_jd=date_to_jd(2026, 9, 22),
                               speed=100.0)

    def tearDown(self):
        matplotlib.pyplot.close(self.sim.fig)

    def step(self, seconds):
        """Advance one frame as though `seconds` of wall time had passed.

        _update() reads the clock again itself, so the elapsed time it sees is
        `seconds` plus the test's own overhead -- hence the deltas below.
        """
        self.sim._last_wall = time.perf_counter() - seconds
        self.sim._update(0)


class TestSimulationSetup(SimulationTestCase):

    def test_an_artist_exists_for_every_body(self):
        for body in ALL_BODIES:
            with self.subTest(body=body.name):
                self.assertIn(body.name, self.sim.body_markers)
                self.assertIn(body.name, self.sim.body_labels)
                self.assertIn(body.name, self.sim.orbit_lines)
                self.assertIn(body.name, self.sim.trail_collections)

    def test_axis_limits_follow_the_view_radius(self):
        radius = self.sim.view_radius
        self.assertAlmostEqual(self.sim.ax.get_xlim3d()[1], radius)
        self.assertAlmostEqual(self.sim.ax.get_zlim3d()[1], radius * main.Z_SQUASH)

    def test_marker_vertices_are_arrays(self):
        # Regression: matplotlib's axlim_clip path reads .shape off the vertex
        # data, so plain lists raise AttributeError partway through a draw.
        self.sim._draw_frame()
        for name, marker in self.sim.body_markers.items():
            with self.subTest(body=name):
                for axis in marker.get_data_3d():
                    self.assertIsInstance(axis, np.ndarray)

    def test_markers_sit_where_the_model_says(self):
        for body in ALL_BODIES:
            with self.subTest(body=body.name):
                expected = warp(body.position(self.sim.jd), self.sim.compressed)
                got = np.array(
                    self.sim.body_markers[body.name].get_data_3d()).ravel()
                np.testing.assert_allclose(got, expected, rtol=1e-9)

    def test_trail_has_one_segment_per_sample_gap(self):
        # get_segments() exposes the projected 2D segments, which only exist
        # once the collection has been drawn.
        self.sim.fig.canvas.draw()
        segments = self.sim.trail_collections["Earth"].get_segments()
        self.assertEqual(len(segments), main.TRAIL_SAMPLES - 1)

    def test_renders_without_error(self):
        self.sim.fig.canvas.draw()


class TestTimeAdvance(SimulationTestCase):

    def test_time_moves_forward_at_the_requested_rate(self):
        start = self.sim.jd
        self.step(0.2)
        self.assertAlmostEqual(self.sim.jd - start, 100.0 * 0.2, delta=0.01)

    def test_long_stall_is_clamped(self):
        start = self.sim.jd
        self.step(10.0)              # a 10 s freeze must not jump 1000 days
        self.assertAlmostEqual(self.sim.jd - start, 100.0 * 0.25, delta=0.01)

    def test_pause_freezes_the_clock(self):
        self.sim.toggle_pause()
        start = self.sim.jd
        self.step(0.2)
        self.assertEqual(self.sim.jd, start)
        self.assertTrue(self.sim.paused)
        self.assertEqual(self.sim.btn_play.label.get_text(), "Play")

    def test_resume_restarts_the_clock(self):
        self.sim.toggle_pause()
        self.sim.toggle_pause()
        start = self.sim.jd
        self.step(0.2)
        self.assertGreater(self.sim.jd, start)
        self.assertEqual(self.sim.btn_play.label.get_text(), "Pause")

    def test_reverse_runs_time_backwards(self):
        self.sim.reverse()
        start = self.sim.jd
        self.step(0.2)
        self.assertAlmostEqual(self.sim.jd - start, -100.0 * 0.2, delta=0.01)

    def test_reverse_twice_is_forward_again(self):
        self.sim.reverse()
        self.sim.reverse()
        self.assertEqual(self.sim.direction, 1.0)

    def test_reset_returns_to_the_start_date(self):
        self.step(0.2)
        self.sim.reset()
        self.assertEqual(self.sim.jd, self.sim.start_jd)

    def test_reset_can_jump_to_a_given_date(self):
        target = date_to_jd(2040, 6, 1)
        self.sim.reset(target)
        self.assertEqual(self.sim.jd, target)

    def test_planets_move_between_frames(self):
        before = np.array(self.sim.body_markers["Mercury"].get_data_3d()).ravel()
        self.step(0.25)
        after = np.array(self.sim.body_markers["Mercury"].get_data_3d()).ravel()
        self.assertGreater(np.linalg.norm(after - before), 1e-3)


class TestSpeedControl(SimulationTestCase):

    def test_speeding_up_and_slowing_down(self):
        self.sim.scale_speed(1.5)
        self.assertAlmostEqual(self.sim.speed, 150.0)
        self.sim.scale_speed(1 / 1.5)
        self.assertAlmostEqual(self.sim.speed, 100.0)

    def test_speed_is_clamped_to_a_usable_range(self):
        for _ in range(50):
            self.sim.scale_speed(1 / 1.5)
        self.assertAlmostEqual(self.sim.speed, 0.1)
        for _ in range(100):
            self.sim.scale_speed(1.5)
        self.assertAlmostEqual(self.sim.speed, 25_000.0)

    def test_slider_tracks_the_speed(self):
        self.sim.scale_speed(1.5)
        self.assertAlmostEqual(self.sim.slider.val,
                               math.log10(self.sim.speed), places=6)

    def test_slider_sets_the_speed_logarithmically(self):
        self.sim._on_slider(3.0)
        self.assertAlmostEqual(self.sim.speed, 1000.0)

    def test_speed_change_does_not_move_time(self):
        start = self.sim.jd
        self.sim.scale_speed(1.5)
        self.assertEqual(self.sim.jd, start)


class TestViewControls(SimulationTestCase):

    def test_toggling_scale_switches_mode_and_view(self):
        self.assertTrue(self.sim.compressed)
        self.sim.toggle_scale()
        self.assertFalse(self.sim.compressed)
        self.assertEqual(self.sim.view_radius, main.DEFAULT_VIEW["real"])
        self.sim.toggle_scale()
        self.assertTrue(self.sim.compressed)
        self.assertEqual(self.sim.view_radius, main.DEFAULT_VIEW["compressed"])

    def test_real_scale_puts_neptune_at_its_true_distance(self):
        self.sim.toggle_scale()
        pos = np.array(self.sim.body_markers["Neptune"].get_data_3d()).ravel()
        self.assertAlmostEqual(float(np.linalg.norm(pos)), 30.0, delta=0.5)

    def test_zoom_changes_the_limits(self):
        before = self.sim.view_radius
        self.sim.zoom(0.5)
        self.assertAlmostEqual(self.sim.view_radius, before * 0.5)
        self.assertAlmostEqual(self.sim.ax.get_xlim3d()[1], before * 0.5)

    def test_zoom_is_clamped(self):
        for _ in range(200):
            self.sim.zoom(0.8)
        self.assertGreaterEqual(self.sim.view_radius, 0.05)
        for _ in range(400):
            self.sim.zoom(1.25)
        self.assertLessEqual(self.sim.view_radius, 400.0)

    def test_scroll_inside_the_plot_zooms(self):
        before = self.sim.view_radius
        self.sim._on_scroll(FakeEvent(button="up", inaxes=self.sim.ax))
        self.assertLess(self.sim.view_radius, before)

    def test_scroll_outside_the_plot_is_ignored(self):
        before = self.sim.view_radius
        self.sim._on_scroll(FakeEvent(button="up", inaxes=None))
        self.assertEqual(self.sim.view_radius, before)

    def test_no_body_is_clipped_at_any_camera_angle(self):
        # Regression: an over-large box_aspect zoom pushed outer planets past
        # the axes' 2D clip box, so they vanished while the camera rotated.
        lost = []
        for elev in (0, 45, 89):
            for azim in (0, 90, 180, 270):
                self.sim.ax.view_init(elev=elev, azim=azim)
                self.sim._draw_frame()
                self.sim.fig.canvas.draw()
                for name, marker in self.sim.body_markers.items():
                    point = self.sim.ax.transData.transform(
                        np.column_stack(marker.get_data()))[0]
                    if not self.sim.ax.bbox.contains(*point):
                        lost.append(f"{name} at elev={elev} azim={azim}")
        self.assertEqual(lost, [])


class TestKeyBindings(SimulationTestCase):

    def press(self, key):
        self.sim._on_key(FakeEvent(key=key))

    def test_space_toggles_pause(self):
        self.press(" ")
        self.assertTrue(self.sim.paused)
        self.press(" ")
        self.assertFalse(self.sim.paused)

    def test_arrow_keys_change_speed(self):
        self.press("right")
        self.assertAlmostEqual(self.sim.speed, 150.0)
        self.press("left")
        self.assertAlmostEqual(self.sim.speed, 100.0)

    def test_visibility_toggles(self):
        for key, attr in (("t", "show_trails"), ("o", "show_orbits"),
                          ("l", "show_labels"), ("h", "show_help")):
            with self.subTest(key=key):
                before = getattr(self.sim, attr)
                self.press(key)
                self.assertNotEqual(getattr(self.sim, attr), before)

    def test_hiding_trails_empties_them(self):
        trail = self.sim.trail_collections["Earth"]
        self.sim.fig.canvas.draw()
        self.assertEqual(len(trail.get_segments()), main.TRAIL_SAMPLES - 1)
        self.press("t")
        self.sim._draw_frame()
        self.sim.fig.canvas.draw()
        self.assertEqual(len(trail.get_segments()), 0)

    def test_hiding_labels_hides_the_sun_label_too(self):
        self.press("l")
        self.sim._draw_frame()
        self.assertFalse(self.sim.body_labels["Earth"].get_visible())
        self.assertFalse(self.sim.sun_label.get_visible())

    def test_hiding_orbits_hides_the_orbit_lines(self):
        self.press("o")
        self.sim._draw_frame()
        self.assertFalse(self.sim.orbit_lines["Earth"].get_visible())

    def test_bracket_keys_zoom(self):
        before = self.sim.view_radius
        self.press("]")
        self.assertLess(self.sim.view_radius, before)
        self.press("[")
        self.assertAlmostEqual(self.sim.view_radius, before)

    def test_b_reverses_time(self):
        self.press("b")
        self.assertEqual(self.sim.direction, -1.0)

    def test_r_resets(self):
        self.step(0.2)
        self.press("r")
        self.assertEqual(self.sim.jd, self.sim.start_jd)

    def test_s_toggles_scale(self):
        self.press("s")
        self.assertFalse(self.sim.compressed)

    def test_unknown_key_is_harmless(self):
        state = (self.sim.jd, self.sim.speed, self.sim.paused)
        self.press("zz")
        self.press(None)
        self.assertEqual((self.sim.jd, self.sim.speed, self.sim.paused), state)


class TestInfoPanel(SimulationTestCase):

    def test_reports_date_speed_and_state(self):
        text = self.sim.info.get_text()
        self.assertIn(format_jd(self.sim.jd), text)
        self.assertIn("RUNNING", text)
        self.assertIn("forward", text)

    def test_reflects_pause_and_reverse(self):
        self.sim.toggle_pause()
        self.sim.reverse()
        text = self.sim.info.get_text()
        self.assertIn("PAUSED", text)
        self.assertIn("reverse", text)

    def test_date_advances_in_the_panel(self):
        self.sim.reset(date_to_jd(2026, 1, 1))
        self.assertIn("2026-01-01", self.sim.info.get_text())
        self.sim.reset(date_to_jd(2027, 3, 15))
        self.assertIn("2027-03-15", self.sim.info.get_text())


class TestAlternateConfigurations(unittest.TestCase):

    def test_starts_in_real_scale_when_asked(self):
        sim = SolarSystem(list(PLANETS), start_jd=main.J2000, speed=1.0,
                          compressed=False)
        try:
            self.assertFalse(sim.compressed)
            self.assertEqual(sim.view_radius, main.DEFAULT_VIEW["real"])
            sim.fig.canvas.draw()
        finally:
            matplotlib.pyplot.close(sim.fig)

    def test_works_without_pluto(self):
        sim = SolarSystem(list(PLANETS), start_jd=main.J2000, speed=1.0)
        try:
            self.assertNotIn("Pluto", sim.body_markers)
            sim.fig.canvas.draw()
        finally:
            matplotlib.pyplot.close(sim.fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
