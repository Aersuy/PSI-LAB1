"""
3D Solar System Simulator
=========================

An interactive, physically-grounded 3D visualisation of the solar system.

Planet positions are computed from real Keplerian orbital elements (JPL
approximate elements for the epoch J2000, with secular rates), so
eccentricities, inclinations and orbital periods are the real ones -- the
orbits are tilted ellipses, not perfect circles, and the planets really do
drift out of the ecliptic plane.

Controls
--------
    drag            rotate the camera
    scroll / [ ]    zoom in and out
    space           pause / resume
    left / right    slow down / speed up (x1.5 per press)
    b               reverse the direction of time
    r               reset to the starting date
    n               jump to today
    s               toggle real vs. distance-compressed scale
    t / o / l       toggle trails / orbit paths / labels
    h               toggle the help overlay
    q               quit

The slider at the bottom sets simulation speed in *days per real second*.

Usage
-----
    python main.py
    python main.py --date 2026-09-08 --speed 30
    python main.py --real-scale --no-pluto
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

J2000 = 2451545.0          # Julian date of the J2000.0 epoch
DAYS_PER_CENTURY = 36525.0
EARTH_RADIUS_KM = 6371.0
SUN_RADIUS_KM = 696_340.0

TRAIL_SAMPLES = 48         # points per motion trail
TRAIL_FRACTION = 0.14      # trail covers this fraction of the orbital period
ORBIT_SAMPLES = 400        # points per drawn orbit ellipse

COMPRESS_EXPONENT = 0.4    # display radius = true radius ** exponent
DEFAULT_VIEW = {"real": 32.0, "compressed": 4.6}
Z_SQUASH = 0.35            # height of the plot box relative to its width


# --------------------------------------------------------------------------- #
# Orbital mechanics
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Body:
    """A planet described by its Keplerian elements at J2000 plus secular rates.

    Angles are in degrees, ``a`` is in astronomical units, and every ``*_dot``
    is the change per Julian century.  Element set: JPL/Standish "Keplerian
    Elements for Approximate Positions of the Major Planets", valid 1800-2050.
    """

    name: str
    color: str
    radius_km: float
    a: float
    a_dot: float
    e: float
    e_dot: float
    inc: float
    inc_dot: float
    mean_long: float          # L, mean longitude
    mean_long_dot: float
    peri_long: float          # varpi, longitude of perihelion
    peri_long_dot: float
    node_long: float          # Omega, longitude of the ascending node
    node_long_dot: float

    @property
    def period_days(self) -> float:
        """Orbital period from Kepler's third law."""
        return 365.256 * self.a ** 1.5

    def elements_at(self, jd):
        """Return (a, e, inclination, argument of perihelion, node, M) in radians."""
        t = (np.asarray(jd, dtype=float) - J2000) / DAYS_PER_CENTURY
        a = self.a + self.a_dot * t
        e = self.e + self.e_dot * t
        inc = np.radians(self.inc + self.inc_dot * t)
        peri_long = self.peri_long + self.peri_long_dot * t
        node = np.radians(self.node_long + self.node_long_dot * t)
        arg_peri = np.radians(peri_long) - node
        mean_long = self.mean_long + self.mean_long_dot * t
        # Mean anomaly, wrapped to [-180, 180) so Newton's method starts close.
        mean_anom = np.radians((mean_long - peri_long + 180.0) % 360.0 - 180.0)
        return a, e, inc, arg_peri, node, mean_anom

    def position(self, jd):
        """Heliocentric ecliptic position in AU; ``jd`` may be a scalar or array."""
        a, e, inc, arg_peri, node, mean_anom = self.elements_at(jd)
        ecc_anom = solve_kepler(mean_anom, e)
        x_orb = a * (np.cos(ecc_anom) - e)
        y_orb = a * np.sqrt(1.0 - e * e) * np.sin(ecc_anom)
        return orbital_to_ecliptic(x_orb, y_orb, arg_peri, node, inc)

    def orbit_path(self, jd, samples: int = ORBIT_SAMPLES):
        """One full ellipse traced with the elements frozen at ``jd``."""
        a, e, inc, arg_peri, node, _ = self.elements_at(jd)
        ecc_anom = np.linspace(0.0, 2.0 * np.pi, samples)
        x_orb = a * (np.cos(ecc_anom) - e)
        y_orb = a * np.sqrt(1.0 - e * e) * np.sin(ecc_anom)
        return orbital_to_ecliptic(x_orb, y_orb, arg_peri, node, inc)


def solve_kepler(mean_anom, ecc, tol: float = 1e-11, max_iter: int = 60):
    """Solve M = E - e*sin(E) for the eccentric anomaly E by Newton-Raphson."""
    mean_anom = np.asarray(mean_anom, dtype=float)
    ecc_anom = mean_anom + ecc * np.sin(mean_anom)
    for _ in range(max_iter):
        delta = (ecc_anom - ecc * np.sin(ecc_anom) - mean_anom) / (
            1.0 - ecc * np.cos(ecc_anom)
        )
        ecc_anom = ecc_anom - delta
        if np.max(np.abs(delta)) < tol:
            break
    return ecc_anom


def orbital_to_ecliptic(x_orb, y_orb, arg_peri, node, inc):
    """Rotate in-plane orbital coordinates into the ecliptic frame."""
    cos_w, sin_w = np.cos(arg_peri), np.sin(arg_peri)
    cos_o, sin_o = np.cos(node), np.sin(node)
    cos_i, sin_i = np.cos(inc), np.sin(inc)

    x = (cos_w * cos_o - sin_w * sin_o * cos_i) * x_orb + (
        -sin_w * cos_o - cos_w * sin_o * cos_i
    ) * y_orb
    y = (cos_w * sin_o + sin_w * cos_o * cos_i) * x_orb + (
        -sin_w * sin_o + cos_w * cos_o * cos_i
    ) * y_orb
    z = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb
    return np.stack([x, y, z], axis=-1)


# --------------------------------------------------------------------------- #
# The bodies
# --------------------------------------------------------------------------- #

PLANETS: list[Body] = [
    Body("Mercury", "#9C9187", 2439.7,
         0.38709927, 0.00000037, 0.20563593, 0.00001906,
         7.00497902, -0.00594749, 252.25032350, 149472.67411175,
         77.45779628, 0.16047689, 48.33076593, -0.12534081),
    Body("Venus", "#E3B778", 6051.8,
         0.72333566, 0.00000390, 0.00677672, -0.00004107,
         3.39467605, -0.00078890, 181.97909950, 58517.81538729,
         131.60246718, 0.00268329, 76.67984255, -0.27769418),
    Body("Earth", "#4A90D9", 6371.0,
         1.00000261, 0.00000562, 0.01671123, -0.00004392,
         -0.00001531, -0.01294668, 100.46457166, 35999.37244981,
         102.93768193, 0.32327364, 0.0, 0.0),
    Body("Mars", "#C1440E", 3389.5,
         1.52371034, 0.00001847, 0.09339410, 0.00007882,
         1.84969142, -0.00813131, -4.55343205, 19140.30268499,
         -23.94362959, 0.44441088, 49.55953891, -0.29257343),
    Body("Jupiter", "#D8A47F", 69911.0,
         5.20288700, -0.00011607, 0.04838624, -0.00013253,
         1.30439695, -0.00183714, 34.39644051, 3034.74612775,
         14.72847983, 0.21252668, 100.47390909, 0.20469106),
    Body("Saturn", "#E3C97E", 58232.0,
         9.53667594, -0.00125060, 0.05386179, -0.00050991,
         2.48599187, 0.00193609, 49.95424423, 1222.49362201,
         92.59887831, -0.41897216, 113.66242448, -0.28867794),
    Body("Uranus", "#8FD8E0", 25362.0,
         19.18916464, -0.00196176, 0.04725744, -0.00004397,
         0.77263783, -0.00242939, 313.23810451, 428.48202785,
         170.95427630, 0.40805281, 74.01692503, 0.04240589),
    Body("Neptune", "#3F62D8", 24622.0,
         30.06992276, 0.00026291, 0.00859048, 0.00005105,
         1.77004347, 0.00035372, -55.12002969, 218.45945325,
         44.96476227, -0.32241464, 131.78422574, -0.00508664),
]

PLUTO = Body("Pluto", "#B8A08A", 1188.3,
             39.48211675, -0.00031596, 0.24882730, 0.00005170,
             17.14001206, 0.00004818, 238.92903833, 145.20780515,
             224.06891629, -0.04062942, 110.30393684, -0.01183482)


# --------------------------------------------------------------------------- #
# Calendar helpers
# --------------------------------------------------------------------------- #

def date_to_jd(year: int, month: int, day: float) -> float:
    """Gregorian calendar date -> Julian date."""
    if month <= 2:
        year -= 1
        month += 12
    century = year // 100
    gregorian = 2 - century + century // 4
    return (math.floor(365.25 * (year + 4716))
            + math.floor(30.6001 * (month + 1))
            + day + gregorian - 1524.5)


def jd_to_date(jd: float) -> tuple[int, int, int]:
    """Julian date -> (year, month, day)."""
    shifted = math.floor(jd + 0.5)
    frac = (jd + 0.5) - shifted
    if shifted < 2299161:
        a = shifted
    else:
        alpha = math.floor((shifted - 1867216.25) / 36524.25)
        a = shifted + 1 + alpha - alpha // 4
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e) + frac
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return int(year), int(month), int(day)


def today_jd() -> float:
    """Julian date for the current UTC day."""
    utc = time.gmtime()
    return date_to_jd(utc.tm_year, utc.tm_mon, utc.tm_mday)


def format_jd(jd: float) -> str:
    year, month, day = jd_to_date(jd)
    era = "" if year > 0 else " BCE"
    return f"{abs(year):04d}-{month:02d}-{day:02d}{era}"


# --------------------------------------------------------------------------- #
# Display scaling
# --------------------------------------------------------------------------- #

def warp(position, compressed: bool):
    """Optionally compress heliocentric distance so all orbits fit one view.

    The direction of every vector is preserved and only its length is remapped
    (r -> r**0.4), which keeps the geometry of the orbits recognisable while
    bringing Mercury at 0.39 AU and Neptune at 30 AU into the same frame.
    """
    if not compressed:
        return position
    radius = np.linalg.norm(position, axis=-1, keepdims=True)
    radius = np.where(radius == 0.0, 1.0, radius)
    return position * radius ** (COMPRESS_EXPONENT - 1.0)


def marker_size(radius_km: float, base: float = 4.6) -> float:
    """Marker diameter in points; cube-rooted so the Sun does not swallow the view."""
    return base * (radius_km / EARTH_RADIUS_KM) ** (1.0 / 3.0)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

class SolarSystem:
    def __init__(self, bodies: list[Body], start_jd: float, speed: float,
                 compressed: bool = True):
        self.bodies = bodies
        self.start_jd = start_jd
        self.jd = start_jd
        self.speed = speed              # days of simulation per real second
        self.direction = 1.0
        self.paused = False
        self.compressed = compressed
        self.show_trails = True
        self.show_orbits = True
        self.show_labels = True
        self.show_help = True
        self.view_radius = DEFAULT_VIEW["compressed" if compressed else "real"]
        self._last_wall = time.perf_counter()

        self._build_figure()
        self._build_artists()
        self._rebuild_orbits()
        self._apply_limits()
        self._draw_frame()

    # -- figure construction ------------------------------------------------ #

    def _build_figure(self) -> None:
        for key in ("keymap.save", "keymap.yscale", "keymap.xscale", "keymap.zoom",
                    "keymap.pan", "keymap.home", "keymap.grid", "keymap.grid_minor",
                    "keymap.fullscreen", "keymap.back", "keymap.forward"):
            plt.rcParams[key] = []

        self.fig = plt.figure(figsize=(11.5, 8.5), facecolor="#05060B")
        self.fig.canvas.manager.set_window_title("3D Solar System")

        # Static starfield painted behind the 3D axes.
        stars = self.fig.add_axes((0, 0, 1, 1), zorder=0)
        stars.set_facecolor("#05060B")
        stars.set_axis_off()
        rng = np.random.default_rng(7)
        n = 900
        stars.scatter(rng.random(n), rng.random(n),
                      s=rng.gamma(1.4, 1.1, n),
                      c="white", alpha=rng.uniform(0.15, 0.8, n),
                      linewidths=0, marker=".")
        stars.set_xlim(0, 1)
        stars.set_ylim(0, 1)

        self.ax = self.fig.add_axes((0.0, 0.10, 1.0, 0.90), projection="3d",
                                    zorder=1)
        self.ax.set_facecolor("none")
        self.ax.patch.set_alpha(0.0)
        self.ax.set_axis_off()
        self.ax.view_init(elev=26, azim=-58)
        # zoom > 1 reclaims the wide margins matplotlib leaves around a 3D box.
        # Above ~1.3 a body sitting at the edge of the view falls outside the
        # axes' own 2D clip box at some camera angles and vanishes mid-rotation,
        # so this stays just under that with room to spare for the marker radius.
        self.ax.set_box_aspect((1.0, 1.0, Z_SQUASH), zoom=1.28)

        self._build_widgets()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _build_widgets(self) -> None:
        style = dict(color="#1B2233", hovercolor="#2C3A55")

        slider_ax = self.fig.add_axes((0.32, 0.050, 0.40, 0.026),
                                      facecolor="#141A28", zorder=3)
        self.slider = Slider(slider_ax, "", -1.0, 4.4,
                             valinit=math.log10(max(self.speed, 0.1)),
                             color="#4A90D9", track_color="#141A28")
        self.slider.valtext.set_visible(False)
        self.slider.on_changed(self._on_slider)
        self.fig.text(0.32, 0.088, "SIMULATION SPEED", color="#5C6B85",
                      fontsize=7.5, family="monospace", zorder=3)

        self.btn_play = Button(
            self.fig.add_axes((0.045, 0.040, 0.075, 0.045), zorder=3),
            "Pause", **style)
        self.btn_play.on_clicked(lambda _: self.toggle_pause())

        self.btn_reset = Button(
            self.fig.add_axes((0.128, 0.040, 0.075, 0.045), zorder=3),
            "Reset", **style)
        self.btn_reset.on_clicked(lambda _: self.reset())

        self.btn_scale = Button(
            self.fig.add_axes((0.211, 0.040, 0.085, 0.045), zorder=3),
            "Scale", **style)
        self.btn_scale.on_clicked(lambda _: self.toggle_scale())

        for button in (self.btn_play, self.btn_reset, self.btn_scale):
            button.label.set_color("#C7D3E8")
            button.label.set_fontsize(9)

        self.btn_slower = Button(
            self.fig.add_axes((0.745, 0.040, 0.040, 0.045), zorder=3),
            "<<", **style)
        self.btn_slower.on_clicked(lambda _: self.scale_speed(1 / 1.5))

        self.btn_faster = Button(
            self.fig.add_axes((0.792, 0.040, 0.040, 0.045), zorder=3),
            ">>", **style)
        self.btn_faster.on_clicked(lambda _: self.scale_speed(1.5))

        self.btn_reverse = Button(
            self.fig.add_axes((0.839, 0.040, 0.055, 0.045), zorder=3),
            "Reverse", **style)
        self.btn_reverse.on_clicked(lambda _: self.reverse())

        for button in (self.btn_slower, self.btn_faster, self.btn_reverse):
            button.label.set_color("#C7D3E8")
            button.label.set_fontsize(8)

        self.info = self.fig.text(0.018, 0.965, "", color="#D7E1F2", fontsize=10.5,
                                  family="monospace", va="top", zorder=3,
                                  linespacing=1.6)
        self.help = self.fig.text(
            0.982, 0.965,
            "drag  rotate\n"
            "scroll / [ ]  zoom\n"
            "space  pause\n"
            "< >  speed     b  reverse\n"
            "r  reset       n  today\n"
            "s  scale       t  trails\n"
            "o  orbits      l  labels\n"
            "h  hide help   q  quit",
            color="#5C6B85", fontsize=8.5, family="monospace",
            va="top", ha="right", zorder=3, linespacing=1.6)

    def _build_artists(self) -> None:
        # Sun: a bright core wrapped in a few translucent haloes.
        for size, alpha in ((marker_size(SUN_RADIUS_KM) * 3.2, 0.06),
                            (marker_size(SUN_RADIUS_KM) * 2.1, 0.10),
                            (marker_size(SUN_RADIUS_KM) * 1.45, 0.18)):
            self.ax.plot([0], [0], [0], marker="o", markersize=size,
                         color="#FFC93C", alpha=alpha, linestyle="none")
        self.ax.plot([0], [0], [0], marker="o",
                     markersize=marker_size(SUN_RADIUS_KM),
                     color="#FFD65C", linestyle="none", zorder=5)
        self.sun_label = self.ax.text(0, 0, 0, "  Sun", color="#FFD65C",
                                      fontsize=8, family="monospace")

        self.orbit_lines = {}
        self.trail_collections = {}
        self.body_markers = {}
        self.body_labels = {}

        fade = np.linspace(0.0, 1.0, TRAIL_SAMPLES - 1) ** 2

        # axlim_clip keeps everything outside the current view box from being
        # drawn, so zooming into the inner system does not leave stray outer
        # orbits and labels smeared across the frame.
        empty = np.empty(0)
        for body in self.bodies:
            (orbit,) = self.ax.plot(empty, empty, empty, color=body.color,
                                    linewidth=0.7, alpha=0.28, axlim_clip=True)
            self.orbit_lines[body.name] = orbit

            rgba = np.zeros((TRAIL_SAMPLES - 1, 4))
            rgba[:, :3] = plt.matplotlib.colors.to_rgb(body.color)
            rgba[:, 3] = fade * 0.85
            trail = Line3DCollection([], colors=rgba, linewidths=1.6,
                                     axlim_clip=True)
            # autolim=False: the collection starts empty and the axis limits are
            # driven by view_radius, not by the data.
            self.ax.add_collection3d(trail, autolim=False)
            self.trail_collections[body.name] = trail

            (marker,) = self.ax.plot(empty, empty, empty, marker="o",
                                     markersize=marker_size(body.radius_km),
                                     color=body.color, linestyle="none",
                                     markeredgecolor="#05060B",
                                     markeredgewidth=0.5, zorder=6,
                                     axlim_clip=True)
            self.body_markers[body.name] = marker

            self.body_labels[body.name] = self.ax.text(
                0, 0, 0, f"  {body.name}", color=body.color, fontsize=8,
                family="monospace", alpha=0.9, axlim_clip=True)

    # -- geometry ----------------------------------------------------------- #

    def _rebuild_orbits(self) -> None:
        for body in self.bodies:
            path = warp(body.orbit_path(self.jd), self.compressed)
            self.orbit_lines[body.name].set_data_3d(path[:, 0], path[:, 1],
                                                    path[:, 2])

    def _apply_limits(self) -> None:
        radius = self.view_radius
        self.ax.set_xlim(-radius, radius)
        self.ax.set_ylim(-radius, radius)
        self.ax.set_zlim(-radius * Z_SQUASH, radius * Z_SQUASH)

    # -- per-frame work ----------------------------------------------------- #

    def _draw_frame(self) -> None:
        for body in self.bodies:
            position = warp(body.position(self.jd), self.compressed)
            marker = self.body_markers[body.name]
            # Slices, not lists: the axlim_clip code path needs real arrays.
            marker.set_data_3d(position[0:1], position[1:2], position[2:3])

            label = self.body_labels[body.name]
            label.set_position_3d(tuple(position))
            label.set_visible(self.show_labels)

            self.orbit_lines[body.name].set_visible(self.show_orbits)

            trail = self.trail_collections[body.name]
            if self.show_trails:
                window = body.period_days * TRAIL_FRACTION * self.direction
                times = self.jd - np.linspace(window, 0.0, TRAIL_SAMPLES)
                points = warp(body.position(times), self.compressed)
                segments = np.stack([points[:-1], points[1:]], axis=1)
                trail.set_segments(segments)
            else:
                trail.set_segments([])

        self.sun_label.set_visible(self.show_labels)
        self._update_info()

    def _update_info(self) -> None:
        state = "PAUSED" if self.paused else "RUNNING"
        arrow = "forward" if self.direction > 0 else "reverse"
        scale = (f"compressed  r^{COMPRESS_EXPONENT:g}"
                 if self.compressed else "true distances")
        self.info.set_text(
            f"DATE    {format_jd(self.jd)}\n"
            f"SPEED   {self.speed:,.1f} days / sec   ({arrow})\n"
            f"STATE   {state}\n"
            f"SCALE   {scale}\n"
            f"VIEW    +/- {self.view_radius:.2f} units"
        )

    def _update(self, _frame) -> None:
        now = time.perf_counter()
        elapsed = min(now - self._last_wall, 0.25)  # clamp after a stall
        self._last_wall = now
        if not self.paused:
            self.jd += self.speed * self.direction * elapsed
        self._draw_frame()

    # -- commands ----------------------------------------------------------- #

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.btn_play.label.set_text("Play" if self.paused else "Pause")
        self._update_info()

    def reset(self, jd: float | None = None) -> None:
        self.jd = self.start_jd if jd is None else jd
        self._rebuild_orbits()
        self._draw_frame()

    def reverse(self) -> None:
        self.direction *= -1.0
        self._update_info()

    def scale_speed(self, factor: float) -> None:
        self.speed = float(np.clip(self.speed * factor, 0.1, 25_000.0))
        self.slider.eventson = False
        self.slider.set_val(math.log10(self.speed))
        self.slider.eventson = True
        self._update_info()

    def toggle_scale(self) -> None:
        self.compressed = not self.compressed
        self.view_radius = DEFAULT_VIEW["compressed" if self.compressed else "real"]
        self._rebuild_orbits()
        self._apply_limits()
        self._draw_frame()

    def zoom(self, factor: float) -> None:
        self.view_radius = float(np.clip(self.view_radius * factor, 0.05, 400.0))
        self._apply_limits()
        self._update_info()

    # -- events ------------------------------------------------------------- #

    def _on_slider(self, value: float) -> None:
        self.speed = 10.0 ** value
        self._update_info()

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        self.zoom(0.88 if event.button == "up" else 1.0 / 0.88)

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == " ":
            self.toggle_pause()
        elif key in ("right", "up", "+", "."):
            self.scale_speed(1.5)
        elif key in ("left", "down", "-", ","):
            self.scale_speed(1 / 1.5)
        elif key == "b":
            self.reverse()
        elif key == "r":
            self.reset()
        elif key == "n":
            self.reset(today_jd())
        elif key == "s":
            self.toggle_scale()
        elif key == "t":
            self.show_trails = not self.show_trails
        elif key == "o":
            self.show_orbits = not self.show_orbits
        elif key == "l":
            self.show_labels = not self.show_labels
        elif key == "h":
            self.show_help = not self.show_help
            self.help.set_visible(self.show_help)
        elif key == "]":
            self.zoom(0.8)
        elif key == "[":
            self.zoom(1.25)

    # -- run ---------------------------------------------------------------- #

    def run(self) -> None:
        self._last_wall = time.perf_counter()
        self.anim = FuncAnimation(self.fig, self._update, interval=16,
                                  blit=False, cache_frame_data=False,
                                  save_count=0)
        plt.show()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive 3D solar system simulator.")
    parser.add_argument("--date", default=None,
                        help="start date as YYYY-MM-DD (default: today)")
    parser.add_argument("--speed", type=float, default=20.0,
                        help="initial speed in simulated days per real second")
    parser.add_argument("--real-scale", action="store_true",
                        help="use true distances instead of compressing them")
    parser.add_argument("--no-pluto", action="store_true",
                        help="leave Pluto out of the simulation")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.date:
        try:
            year, month, day = (int(part) for part in args.date.split("-"))
        except ValueError as exc:
            raise SystemExit(f"could not read --date {args.date!r}; "
                             "expected YYYY-MM-DD") from exc
        start = date_to_jd(year, month, day)
    else:
        start = today_jd()

    bodies = list(PLANETS) if args.no_pluto else [*PLANETS, PLUTO]
    sim = SolarSystem(bodies, start_jd=start,
                      speed=max(args.speed, 0.1),
                      compressed=not args.real_scale)
    sim.run()


if __name__ == "__main__":
    main()
