"""
===============================================================================
ZENITHS
Zenith Exposure and Nighttime Intelligent Target Handling & Scheduling
===============================================================================
Visibility-aware astronomical observation scheduler.

Dependencies:  pip install astropy numpy pandas matplotlib pulp highspy tzdata

Features
--------
  - Alt/Az tracking for target lists from a named observatory (observatories.json)
  - Local-time observing windows with timezone-aware UTC conversion
  - Hard visibility constraints from altitude, airmass, twilight and Moon separation
  - Smooth observing-efficiency score using brightness, airmass, twilight and Moon terms
  - Three-stage scheduling: complete targets, fill gaps, then globally rebalance separators
  - Per-target exposure requirements with a command-line fallback exposure time
  - Minimum continuous observing-window and target-switch overhead constraints
  - Multi-night optimization with exact sampled twilight-to-twilight schedule anchoring
  - TSV observation report and optional multi-page PDF night charts
===============================================================================
"""

from __future__ import annotations

import os
import re
import colorsys
import json
import math
import sys
import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Optional

APP_NAME = "ZENITHS"
APP_EXPANSION = "Zenith Exposure and Nighttime Intelligent Target Handling & Scheduling"
APP_TAGLINE = "Visibility-aware astronomical observation scheduler"
__version__ = "0.1.0"

# ZoneInfo: stdlib since Python 3.9; backport via `pip install tzdata` for 3.8
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # type: ignore
    except ImportError:
        sys.exit(
            "\n[ERROR] zoneinfo is required.\n"
            "Python ≥ 3.9 includes it in the stdlib.\n"
            "For Python 3.8 install the backport:  pip install tzdata\n"
        )

try:
    import pulp
except ImportError:
    sys.exit(
        "\n[ERROR] pulp and its dependencies are required for scheduling.\n"
        "Install with:  pip install pulp\n"
    )

# -- main imports ----------------------------------------------------------
try:
    import pandas as pd
    import numpy as np
    from astropy import units as au
    from astropy.coordinates import (
        AltAz,
        EarthLocation,
        SkyCoord,
        get_body,
        solar_system_ephemeris,
    )
    from astropy.time import Time
except ImportError:
    sys.exit(
        "\n[ERROR] astropy, pandas, numpy are required.\n"
        "Install with:  pip install astropy pandas numpy\n"
    )

# Optional matplotlib
try:
    import matplotlib.ticker as mticker
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class Observatory:
    """
    Observer location on Earth.

    Parameters
    ----------
    name      : Human-readable site name.
    latitude  : Decimal degrees, positive = North, negative = South.
    longitude : Decimal degrees, positive = East, negative = West.
    elevation : Metres above sea level (default 0).
    timezone  : IANA timezone name (e.g. "Australia/Sydney", "Pacific/Honolulu",
                "UTC").  ZoneInfo resolves the correct UTC offset — including
                Daylight Saving Time — at the exact moment of each conversion,
                so there is no risk of an hour-long error on DST transition
                nights.  Use the IANA database name, NOT a numeric offset string.
    """
    name: str
    latitude: float
    longitude: float
    elevation: float = 0.0
    timezone: str = "UTC"           # IANA tz name, e.g. "Australia/Sydney"

    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be in [-90, 90], got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be in [-180, 180], got {self.longitude}")
        # Validate the timezone string immediately so the error is caught early.
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, KeyError):
            raise ValueError(
                f"Unknown IANA timezone {self.timezone!r}. "
                "Check https://en.wikipedia.org/wiki/List_of_tz_database_time_zones"
            )

    @property
    def tz(self) -> ZoneInfo:
        """The ZoneInfo object for this site's timezone."""
        return ZoneInfo(self.timezone)

    def earth_location(self) -> EarthLocation:
        """Return the astropy EarthLocation for this site."""
        return EarthLocation(
            lat=self.latitude * au.deg,
            lon=self.longitude * au.deg,
            height=self.elevation * au.m,
        )

    def local_to_utc(self, local_dt: datetime) -> datetime:
        """
        Interpret a naive local wall-clock datetime as this site's local time
        and return a UTC-aware datetime.

        The fold=0 convention is used when the wall-clock time is ambiguous
        (i.e. clocks fall back and the hour repeats): the first occurrence
        (still in summer/DST time) is taken.
        """
        if local_dt.tzinfo is None:
            aware_local = local_dt.replace(tzinfo=self.tz, fold=0)
        else:
            assert local_dt.tzinfo == self.tz, "local_dt has tzinfo but does not match the observatory timezone"
            aware_local = local_dt
        return aware_local.astimezone(timezone.utc)

    def utc_to_local(self, utc_dt: datetime) -> datetime:
        """Convert a UTC-aware datetime to a naive local wall-clock datetime."""
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        return utc_dt.astimezone(self.tz).replace(tzinfo=None)

    def utc_offset_at(self, utc_dt: datetime) -> timedelta:
        """Return the UTC offset (as timedelta) in effect at *utc_dt*."""
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        return utc_dt.astimezone(self.tz).utcoffset()

    def tz_label_at(self, utc_dt: datetime) -> str:
        """
        Return a compact offset label such as '+10:00' or '+10:30' for display.
        Used wherever the old UTC{sign}{utc_off:g}h label appeared.
        """
        offset = self.utc_offset_at(utc_dt)
        total_minutes = int(offset.total_seconds() // 60)
        sign  = "+" if total_minutes >= 0 else "-"
        total_minutes = abs(total_minutes)
        h, m  = divmod(total_minutes, 60)
        return f"UTC{sign}{h:02d}:{m:02d}" if m else f"UTC{sign}{h:02d}"

@dataclass
class SkySource:
    """A deep-sky target with decimal-degree coordinates and a required exposure time."""

    name: str = None
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    exptime: Optional[float] = None
    magnitude: Optional[float] = None
    min_altitude: float = 15.0

    def __post_init__(self):
        if self.ra_deg is None or self.dec_deg is None:
            raise ValueError(f"Supply RA/Dec in decimal degrees for '{self.name}'.")
        if not isinstance(self.ra_deg, (int, float)) or not 0 <= float(self.ra_deg) < 360:
            raise ValueError(f"ra_deg must be a number in [0, 360), got {self.ra_deg}")
        if not isinstance(self.dec_deg, (int, float)) or not -90 <= float(self.dec_deg) <= 90:
            raise ValueError(f"dec_deg must be a number in [-90, 90], got {self.dec_deg}")
        if self.exptime is not None and self.exptime <= 0:
            raise ValueError(f"exptime must be > 0 minutes, got {self.exptime}")

        self.skycoord = SkyCoord(ra=self.ra_deg * au.deg, dec=self.dec_deg * au.deg, frame="icrs")
        self.ra_hms = self.skycoord.ra.to_string(unit=au.hour, sep="hms", precision=2)
        self.dec_dms = self.skycoord.dec.to_string(unit=au.deg, sep="dms", precision=1, alwayssign=True)

        if self.name is None or self.name.strip() == "":
            self.name = "J" + self.ra_hms + self.dec_dms

@dataclass
class ScheduledBlock:
    """
    One contiguous on-target observation slot assigned to a single source.

    Attributes
    ----------
    source        : The target being observed.
    start_time    : When the telescope is on-target (after slew overhead).
    end_time      : When the observation ends.
    score         : Total visibility score across the block.
    """
    source:      SkySource
    start_time:  Time
    end_time:    Time
    score:       float

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def compute_airmass(altitude_deg: float | np.ndarray) -> float | np.ndarray:
    """
    Airmass using the Rozenberg (1966) formula, accurate to the horizon.
    Returns 99.0 for below-horizon altitudes.
    """
    if isinstance(altitude_deg, np.ndarray):
        airmass = np.full_like(altitude_deg, fill_value=99.0, dtype=np.float64)
        above_horizon = altitude_deg > 0.0
        cos_z = np.cos(np.deg2rad(90.0 - altitude_deg[above_horizon]))
        airmass[above_horizon] = 1.0 / (cos_z + 0.025 * np.exp(-11.0 * cos_z))
        return airmass
    
    if altitude_deg <= 0.0:
        return 99.0
    z     = math.radians(90.0 - altitude_deg)
    cos_z = math.cos(z)
    return 1.0 / (cos_z + 0.025 * math.exp(-11.0 * cos_z))

def compute_moon_illumination(sun_altazs: AltAz, moon_altazs: AltAz) -> tuple[np.ndarray, np.ndarray]:
    """
    Moon phase angle and illumination fraction at *obs_time*.

    Returns
    -------
    (phase_angle_deg, illumination_fraction)
    """
    elongation   = moon_altazs.separation(sun_altazs).deg
    phase_angle  = (180.0 - elongation) % 360.0
    illumination = (1.0 + np.cos(np.deg2rad(phase_angle))) / 2.0
    return phase_angle, np.clip(illumination, 0.0, 1.0)

def phase_name_from(phase_angle_deg: float, illumination: float) -> str:
    pa = phase_angle_deg % 360.0
    k  = illumination
    if k < 0.03:
        return "New Moon"
    if k > 0.97:
        return "Full Moon"
    if pa < 180:
        if k < 0.30:
            return "Waxing Crescent"
        if k < 0.55:
            return "First Quarter"
        return "Waxing Gibbous"
    else:
        if k > 0.55:
            return "Waning Gibbous"
        if k > 0.30:
            return "Last Quarter"
        return "Waning Crescent"

def parse_local_datetime(s: str) -> datetime:
    """Parse an ISO-8601 local datetime string into a naive datetime."""
    s = s.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse datetime {s!r}. "
        "Expected ISO-8601, e.g. '2025-06-15T18:00' or '2025-06-15 18:00:00'."
    )

def get_dark_intervals(
    start_utc: datetime,
    end_utc: datetime,
    location,
    step_minutes: float = 1.0,
    sun_alt_limit: float = -18.0,
) -> list[tuple[datetime, datetime]]:
    """
    Return every contiguous dark interval within [start_utc, end_utc].
    "Dark" means the Sun's altitude is below *sun_alt_limit* degrees.
    """
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)

    step    = timedelta(minutes=step_minutes)
    n_steps = int((end_utc - start_utc) / step) + 1
    times   = [start_utc + step * i for i in range(n_steps)]
    if times[-1] < end_utc:
        times.append(end_utc)

    astropy_times = Time(times, scale="utc")
    altaz_frame   = AltAz(obstime=astropy_times, location=location)

    with solar_system_ephemeris.set("builtin"):
        sun_altaz = get_body("sun", astropy_times, location).transform_to(altaz_frame)

    sun_alts = sun_altaz.alt.deg

    intervals: list[tuple[datetime, datetime]] = []
    in_dark    = False
    dark_start: Optional[datetime] = None

    for i, (t, alt) in enumerate(zip(times, sun_alts)):
        is_dark = float(alt) < sun_alt_limit
        if is_dark and not in_dark:
            dark_start = t
            in_dark    = True
        elif not is_dark and in_dark:
            intervals.append((dark_start, times[i - 1]))
            in_dark = False

    if in_dark and dark_start is not None:
        intervals.append((dark_start, times[-1]))

    return intervals

# -----------------------------------------------------------------------------
# ZENITHS visibility and scheduling engine
# -----------------------------------------------------------------------------

class VisibilityCalculator:
    """Core ZENITHS visibility engine and observation scheduler."""
    def __init__(
        self,
        observatory: Observatory,
        sources: list[SkySource],
        min_altitude: float = 15.0,
        moon_exclusion_deg: float = 30.0,
        max_airmass: float = 3.0,
        bright_time: bool = False,
    ):
        self.obs            = observatory
        self.sources        = sources
        self.location       = observatory.earth_location()
        self.min_altitude   = min_altitude
        self.moon_excl      = moon_exclusion_deg
        self.max_airmass    = max_airmass
        self.bright_time    = bright_time

        # Relative observing-efficiency proxy used by the scheduler. The constants below are intentionally modest:
        # they provide smooth ranking rather than pretending to be a full MOSFIRE exposure-time calculator.
        magnitudes = [src.magnitude for src in sources if src.magnitude is not None and np.isfinite(src.magnitude)]
        self.score_scale = 100.0
        self.mag_reference = float(np.median(magnitudes)) if magnitudes else None
        self.mag_eff_min = 0.10
        self.mag_eff_max = 10.0
        self.extinction_mag_per_airmass = 0.10
        self.sun_soft_range_deg = 6.0
        self.sun_edge_efficiency = 0.65
        self.moon_background_strength = 2.0

        self.__nightly_data = None
        self.__observation_sequences = None

    # -- internal helpers -----------------------------------------------------

    def _to_astropy_time(self, utc: datetime) -> Time:
        if utc.tzinfo is None:
            utc = utc.replace(tzinfo=timezone.utc)
        return Time(utc, scale="utc")

    def _altaz_frame(self, obs_time: Time) -> AltAz:
        return AltAz(obstime=obs_time, location=self.location)


    # -- visibility scoring ---------------------------------------------------
    def _calc_scores_for_source(
        self, source: SkySource, src_altazs: AltAz, moon_altazs: AltAz,
        sun_altazs: AltAz, sun_alt_limit: float = -18.0,
    ) -> np.ndarray:
        """
        Relative observing-efficiency score.

        The score is a positive proxy for useful signal/S/N accumulation per unit time:

            score = 100 * brightness_eff * airmass_eff * sun_eff * moon_eff

        It is not an exposure-time calculator. Its purpose is to rank otherwise legal slots and to make
        integrated score a useful scheduling currency: a difficult/faint source accumulates score more slowly
        and therefore naturally receives more clock time when Stage 3 balances integrated scores.
        """

        min_alt = max(self.min_altitude, source.min_altitude)
        altitudes = np.asarray(src_altazs.alt.deg, dtype=float)
        sun_altitudes = np.asarray(sun_altazs.alt.deg, dtype=float)
        moon_altitudes = np.asarray(moon_altazs.alt.deg, dtype=float)
        air_masses = np.asarray(compute_airmass(altitudes), dtype=float)
        scores = np.full(len(src_altazs), np.nan, dtype=np.float64)

        cons1 = altitudes <= 0.0
        cons2 = altitudes < min_alt
        cons3 = air_masses > self.max_airmass
        cons4 = np.zeros(len(src_altazs), dtype=bool) if self.bright_time else (
            (src_altazs.separation(moon_altazs).deg < self.moon_excl) & (moon_altitudes > 0.0)
        )
        cons5 = sun_altitudes > sun_alt_limit
        observable = ~(cons1 | cons2 | cons3 | cons4 | cons5)

        if not np.any(observable):
            return scores

        # Airmass efficiency: atmospheric extinction plus a soft 1/X degradation for the longer sky path,
        # increased background and seeing. This is deliberately heuristic rather than an instrument ETC.
        x = air_masses[observable]
        transmission_eff = 10.0 ** (-0.4 * self.extinction_mag_per_airmass * (x - 1.0))
        airmass_eff = transmission_eff / x

        # Smooth twilight penalty. The selected twilight altitude remains a hard boundary, but slots just
        # inside it are down-weighted and recover smoothly to full-dark efficiency over the next 6 degrees.
        sun_depth = np.clip((sun_alt_limit - sun_altitudes[observable]) / self.sun_soft_range_deg, 0.0, 1.0)
        sun_smooth = 0.5 - 0.5 * np.cos(np.pi * sun_depth)
        sun_eff = self.sun_edge_efficiency + (1.0 - self.sun_edge_efficiency) * sun_smooth

        # Brightness efficiency. Broad-band magnitude is only a proxy for spectroscopic difficulty, so use
        # a tempered flux-like 10^(-0.4 dm) scaling rather than a much steeper formal flux-squared relation.
        mag_eff = 1.0
        if source.magnitude is not None and self.mag_reference is not None:
            mag_eff = 10.0 ** (-0.4 * (source.magnitude - self.mag_reference))
            mag_eff = float(np.clip(mag_eff, self.mag_eff_min, self.mag_eff_max))

        # Moon penalty is also multiplicative so scores remain positive. In --bright-time mode the Moon is
        # intentionally ignored, exactly as requested by that flag.
        moon_eff = np.ones(np.count_nonzero(observable), dtype=float)
        if not self.bright_time:
            obs_idx = np.flatnonzero(observable)
            moon_up = moon_altitudes[obs_idx] > 0.0
            if np.any(moon_up):
                sep = np.asarray(src_altazs.separation(moon_altazs).deg, dtype=float)[obs_idx]
                sep_fraction = np.clip((sep - self.moon_excl) / (180.0 - self.moon_excl), 0.0, 1.0)
                proximity = (1.0 - sep_fraction) ** 2
                _, illum = compute_moon_illumination(sun_altazs, moon_altazs)
                illum = np.asarray(illum, dtype=float)[obs_idx]
                altitude_factor = np.sqrt(np.sin(np.deg2rad(np.clip(moon_altitudes[obs_idx], 0.0, 90.0))))
                moon_load = illum * altitude_factor * proximity
                moon_eff = 1.0 / (1.0 + self.moon_background_strength * moon_load)

        scores[observable] = self.score_scale * mag_eff * airmass_eff * sun_eff * moon_eff
        return scores

    # -- public API -----------------------------------------------------------
    def generate_tracking_data(
        self,
        start_utc: datetime,
        end_utc: datetime,
        step_minutes: float = 5.0,
        sun_alt_limit: float = -18.0,
    ) -> None:
        """
        Find all observing sessions across every dark night in [start_utc, end_utc].
        """
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)
        if end_utc.tzinfo is None:
            end_utc = end_utc.replace(tzinfo=timezone.utc)
 
        dark_intervals = get_dark_intervals(
            start_utc=start_utc, end_utc=end_utc,
            location=self.location, step_minutes=max(step_minutes, 5.0),
            sun_alt_limit=sun_alt_limit,
        )
 
        if not dark_intervals:
            raise ValueError(
                f"No astronomical dark time (Sun < {sun_alt_limit}°) found between "
                f"{start_utc.strftime('%Y-%m-%d %H:%M')} UTC and "
                f"{end_utc.strftime('%Y-%m-%d %H:%M')} UTC "
                f"for observatory '{self.obs.name}'."
            )
 
        print(f"[INFO] {len(dark_intervals)} dark night(s) found in the requested range.")
        mag_text = "disabled" if self.mag_reference is None else f"median reference {self.mag_reference:.2f} mag"
        print(
            f"[INFO] Visibility score: relative observing-efficiency proxy | magnitude={mag_text} | "
            f"airmass extinction={self.extinction_mag_per_airmass:.2f} mag/airmass"
        )
        print(
            f"[INFO] Twilight soft penalty: efficiency {self.sun_edge_efficiency:.2f} at Sun={sun_alt_limit:.1f} deg "
            f"-> 1.00 by Sun={sun_alt_limit - self.sun_soft_range_deg:.1f} deg"
        )
        if self.bright_time:
            print("[INFO] Moon term disabled by --bright-time.")
        print("[INFO] Getting ready to track sources", end='\r')

        step = timedelta(minutes=step_minutes)
        n_steps = int((end_utc - start_utc) / step) + 1
        times_utc = [start_utc + i * step for i in range(n_steps)]
        if times_utc[-1] < end_utc:
            times_utc.append(end_utc)
        times_utc = np.array(times_utc)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        astropy_times = Time(times_utc, scale="utc")
        frames = AltAz(obstime=astropy_times, location=self.location)
        with solar_system_ephemeris.set("builtin"):
            moon_body = get_body("moon", astropy_times, self.location)
            sun_body  = get_body("sun",  astropy_times, self.location)
        moon_altazs = moon_body.transform_to(frames)
        sun_altazs  = sun_body.transform_to(frames)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("[INFO] Tracking sources across all nights …")
 
        pad = timedelta(minutes=60)
        self.__nightly_data = []
        for night_index, (dark_start, dark_end) in enumerate(dark_intervals):
            duration_h = (dark_end - dark_start).total_seconds() / 3600.0
            start_time = max(dark_start - pad, start_utc)
            end_time   = min(dark_end   + pad, end_utc)
            night_mask = (
                (astropy_times >= self._to_astropy_time(start_time))
                & (astropy_times <= self._to_astropy_time(end_time))
            )
            moon_altaz = moon_altazs[night_mask]
            sun_altaz  = sun_altazs[night_mask]

            print(
                f"[INFO] Night {night_index + 1}/{len(dark_intervals)} "
                f"{dark_start.strftime('%Y-%m-%d %H:%M')} → "
                f"{dark_end.strftime('%H:%M')} UTC ({duration_h:.1f} h)"
            )

            allsrc_altazs = {} 
            allsrc_scores = {}
            for src in self.sources:
                print(f"[INFO] Tracking Source: {src.name}", end='\r')
                src_altaz  = src.skycoord.transform_to(frames[night_mask])
                scores = self._calc_scores_for_source(src, src_altaz, moon_altaz, sun_altaz, sun_alt_limit)
                
                allsrc_altazs[src.name] = src_altaz 
                allsrc_scores[src.name] = scores

            self.__nightly_data.append({
                "astropy_times": astropy_times[night_mask], 
                "src_altazs": allsrc_altazs, 
                "scores": allsrc_scores, 
                "moon": moon_altaz, 
                "sun": sun_altaz,
            })
        
        print(f"[INFO] Tracking complete across all nights. Ready to build observation plans.")

    # -----------------------------------------------------------------------------
    # Observation planning and scheduling
    # -----------------------------------------------------------------------------
  
    def build_observation_plan(
        self, exptime_minutes, min_window_minutes, overhead_minutes, step_minutes,
        max_nights, sun_alt_limit=-18.0,
    ) -> None:

        if max_nights is None:
            raise ValueError("max_nights must be specified")
        if self.__nightly_data is None:
            raise RuntimeError("Run generate_tracking_data() before build_observation_plan().")
        if exptime_minutes <= 0 or min_window_minutes <= 0 or step_minutes <= 0:
            raise ValueError("exptime_minutes, min_window_minutes and step_minutes must be positive")
        if overhead_minutes < 0:
            raise ValueError("overhead_minutes cannot be negative")

        src_names = [src.name for src in self.sources]
        source_map = {src.name: src for src in self.sources}
        night_ids = list(range(len(self.__nightly_data)))
        overhead_steps = int(math.ceil(overhead_minutes / step_minutes))
        min_window_steps = int(math.ceil(min_window_minutes / step_minutes))

        if "HiGHS" not in pulp.listSolvers(onlyAvailable=True):
            raise RuntimeError("HiGHS is not available. Install it with: pip install highspy")

        nthreads = os.cpu_count() or 1
        solver = pulp.HiGHS(msg=False, threads=nthreads, parallel="on", gapRel=1e-6)

        # --------------------------------------------------------------------------------------------------
        # Per-source required exposure. Table values override the command-line fallback.
        # --------------------------------------------------------------------------------------------------
        exptime_steps = {}
        for src in self.sources:
            minutes = src.exptime if src.exptime is not None else exptime_minutes
            if minutes < min_window_minutes:
                raise ValueError(
                    f"{src.name}: exptime={minutes:g} min is shorter than the minimum continuous "
                    f"window of {min_window_minutes:g} min."
                )

            exptime_steps[src.name] = int(math.ceil(minutes / step_minutes))

        # --------------------------------------------------------------------------------------------------
        # Cache slot visibility, integrated slot scores and twilight boundaries for every night/source.
        # Slots use the half-open convention [t_i, t_{i+1}).
        # --------------------------------------------------------------------------------------------------
        night_bounds = {}
        slot_scores = {}
        slot_valid = {}

        for nid, night_data in enumerate(self.__nightly_data):
            times = night_data["astropy_times"]
            sun_alt = np.asarray(night_data["sun"].alt.deg, dtype=float)

            if len(times) < 2:
                continue

            twilight_valid = (sun_alt[:-1] <= sun_alt_limit) & (sun_alt[1:] <= sun_alt_limit)
            twilight_slots = np.flatnonzero(twilight_valid)
            if not len(twilight_slots):
                continue

            night_start = int(twilight_slots[0])
            night_end = int(twilight_slots[-1]) + 1
            night_bounds[nid] = (night_start, night_end)

            for src in self.sources:
                scores = np.asarray(night_data["scores"][src.name], dtype=float)
                values = 0.5 * (scores[:-1] + scores[1:])
                slot_scores[nid, src.name] = values
                slot_valid[nid, src.name] = np.isfinite(values) & twilight_valid

            t0 = self.obs.utc_to_local(times[night_start].datetime)
            t1 = self.obs.utc_to_local(times[night_end].datetime)
            print(
                f"[INFO] Night {nid + 1}: usable {t0.strftime('%H:%M')}–{t1.strftime('%H:%M')} "
                f"(Sun <= {sun_alt_limit:.1f} deg)"
            )

        # --------------------------------------------------------------------------------------------------
        # PHASE 1: one required-exptime core window per source.
        # Stage 1A maximizes the number of completed sources; Stage 1B then maximizes total visibility score.
        # Used nights are anchored exactly to the sampled evening and morning twilight boundaries.
        # --------------------------------------------------------------------------------------------------
        all_blocks = []
        block_id = 0

        for nid in night_ids:
            if nid not in night_bounds:
                continue

            night_start, night_end = night_bounds[nid]

            for src in self.sources:
                nstep = exptime_steps[src.name]
                valid = slot_valid[nid, src.name]
                values = slot_scores[nid, src.name]

                for start_idx in range(night_start, night_end - nstep + 1):
                    end_idx = start_idx + nstep
                    if not np.all(valid[start_idx:end_idx]):
                        continue

                    all_blocks.append({
                        "id": block_id,
                        "source": src.name,
                        "src_obj": src,
                        "night_id": nid,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "score": float(np.sum(values[start_idx:end_idx])),
                        "core": True,
                    })
                    block_id += 1

        if not all_blocks:
            print("[WARN] No feasible required-exptime observing blocks found.")
            self.__observation_sequences = []
            return

        blocks_by_source = {name: [] for name in src_names}
        blocks_by_night = {nid: [] for nid in night_ids}
        for block in all_blocks:
            blocks_by_source[block["source"]].append(block)
            blocks_by_night[block["night_id"]].append(block)

        prob = pulp.LpProblem("required_exptime_schedule", pulp.LpMaximize)
        z = {block["id"]: pulp.LpVariable(f"z_{block['id']}", cat="Binary") for block in all_blocks}
        y = {nid: pulp.LpVariable(f"night_{nid}", cat="Binary") for nid in night_ids}

        for name, blocks in blocks_by_source.items():
            if blocks:
                prob += pulp.lpSum(z[block["id"]] for block in blocks) <= 1

        for nid in night_ids:
            blocks = blocks_by_night[nid]
            if not blocks or nid not in night_bounds:
                prob += y[nid] == 0
                continue

            nslots = len(self.__nightly_data[nid]["astropy_times"]) - 1
            coverage = [[] for _ in range(nslots)]

            for block in blocks:
                stop = min(block["end_idx"] + overhead_steps, nslots)
                for t in range(block["start_idx"], stop):
                    coverage[t].append(block["id"])

            for ids in coverage:
                if len(ids) > 1:
                    prob += pulp.lpSum(z[bid] for bid in ids) <= 1

            for block in blocks:
                prob += z[block["id"]] <= y[nid]
            prob += y[nid] <= pulp.lpSum(z[block["id"]] for block in blocks)

            night_start, night_end = night_bounds[nid]
            edge_start = [z[b["id"]] for b in blocks if b["start_idx"] == night_start]
            edge_end = [z[b["id"]] for b in blocks if b["end_idx"] == night_end]

            if edge_start:
                prob += pulp.lpSum(edge_start) == y[nid]
            else:
                prob += y[nid] == 0

            if edge_end:
                prob += pulp.lpSum(edge_end) == y[nid]
            else:
                prob += y[nid] == 0

        prob += pulp.lpSum(y[nid] for nid in night_ids) <= max_nights

        target_count = pulp.lpSum(z[block["id"]] for block in all_blocks)
        visibility_score = pulp.lpSum(block["score"] * z[block["id"]] for block in all_blocks)

        print(f"[INFO] Stage 1A: maximizing completed sources with HiGHS ({nthreads} threads) …")
        prob.setObjective(target_count)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        if status not in ("Optimal", "Feasible"):
            raise RuntimeError(f"Stage 1A failed with status: {status}")

        best_targets = int(round(pulp.value(target_count) or 0))
        if best_targets == 0:
            raise RuntimeError(
                "No source can be scheduled while also anchoring the used night(s) to both twilight boundaries."
            )

        prob += target_count == best_targets
        print(f"[INFO] Stage 1A: {best_targets}/{len(src_names)} sources completed")
        print("[INFO] Stage 1B: maximizing total visibility score of the required-exptime windows …")

        prob.setObjective(visibility_score)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        if status not in ("Optimal", "Feasible"):
            raise RuntimeError(f"Stage 1B failed with status: {status}")

        night_selected = {nid: [] for nid in night_ids}
        for block in all_blocks:
            if (pulp.value(z[block["id"]]) or 0.0) > 0.5:
                night_selected[block["night_id"]].append({
                    "source": block["source"],
                    "src_obj": block["src_obj"],
                    "night_id": block["night_id"],
                    "start_idx": block["start_idx"],
                    "end_idx": block["end_idx"],
                    "score": block["score"],
                    "core": True,
                })

        selected_sources = {
            block["source"] for blocks in night_selected.values() for block in blocks
        }
        print(
            f"[INFO] Stage 1 solution: {len(selected_sources)}/{len(src_names)} completed sources across "
            f"{sum(bool(blocks) for blocks in night_selected.values())} night(s)"
        )

        # --------------------------------------------------------------------------------------------------
        # PHASE 2: fill internal gaps and absorb residual idle time.
        #
        # The Stage-1 source set is fixed. Stage 2 may add extra windows only for already-completed sources.
        # It first inserts the highest-integrated-score legal filler window (minimum min_window), then extends
        # neighbouring windows into any residual idle interval while preserving the requested switch overhead.
        # --------------------------------------------------------------------------------------------------
        def recalc_score(nid, block):
            values = slot_scores[nid, block["source"]]
            block["score"] = float(np.sum(values[block["start_idx"]:block["end_idx"]]))

        def merge_touching(nid, blocks):
            blocks.sort(key=lambda b: b["start_idx"])
            merged = []

            for block in blocks:
                if merged and merged[-1]["source"] == block["source"] and merged[-1]["end_idx"] == block["start_idx"]:
                    merged[-1]["end_idx"] = block["end_idx"]
                    merged[-1]["core"] = merged[-1]["core"] or block["core"]
                    recalc_score(nid, merged[-1])
                else:
                    merged.append(block)

            return merged

        def best_window_in_gap(nid, left, right):
            best = None

            for name in selected_sources:
                left_overhead = 0 if name == left["source"] else overhead_steps
                right_overhead = 0 if name == right["source"] else overhead_steps
                lo = left["end_idx"] + left_overhead
                hi = right["start_idx"] - right_overhead

                if hi - lo < min_window_steps:
                    continue

                valid = slot_valid[nid, name]
                values = slot_scores[nid, name]
                prefix = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(values), values, 0.0))))

                for start_idx in range(lo, hi - min_window_steps + 1):
                    if not valid[start_idx]:
                        continue

                    max_end = start_idx
                    while max_end < hi and valid[max_end]:
                        max_end += 1

                    if max_end - start_idx < min_window_steps:
                        continue

                    for end_idx in range(start_idx + min_window_steps, max_end + 1):
                        score = float(prefix[end_idx] - prefix[start_idx])
                        duration = end_idx - start_idx
                        rank = (score, duration)

                        if best is None or rank > best["rank"]:
                            best = {
                                "source": name,
                                "src_obj": source_map[name],
                                "night_id": nid,
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                                "score": score,
                                "core": False,
                                "rank": rank,
                            }

            if best is not None:
                best.pop("rank")
            return best

        def fill_night(nid, blocks):
            blocks = merge_touching(nid, blocks)
            night_start, night_end = night_bounds[nid]
            times = self.__nightly_data[nid]["astropy_times"]
            available = night_end - night_start
            initial_windows = len(blocks)
            initial_on_target = sum(block["end_idx"] - block["start_idx"] for block in blocks)
            initial_util = 100.0 * initial_on_target / available if available else 0.0

            print(
                f"[INFO] Stage 2 Night {nid + 1}: start with {initial_windows} core window(s), "
                f"on-target={initial_on_target * step_minutes:.0f} min ({initial_util:.1f}%)"
            )

            added_windows = 0
            while True:
                best = None
                best_index = None

                for i in range(len(blocks) - 1):
                    candidate = best_window_in_gap(nid, blocks[i], blocks[i + 1])
                    if candidate is None:
                        continue

                    rank = (candidate["score"], candidate["end_idx"] - candidate["start_idx"])
                    if best is None or rank > best["rank"]:
                        best = {"block": candidate, "rank": rank}
                        best_index = i + 1

                if best is None:
                    break

                filler = best["block"]
                t0 = self.obs.utc_to_local(times[filler["start_idx"]].datetime)
                t1 = self.obs.utc_to_local(times[filler["end_idx"]].datetime)
                duration = (filler["end_idx"] - filler["start_idx"]) * step_minutes
                mean_rate = filler["score"] / max(filler["end_idx"] - filler["start_idx"], 1)
                print(
                    f"[INFO] Stage 2 Night {nid + 1}: filler {added_windows + 1}: {filler['source']} "
                    f"{t0.strftime('%H:%M')}–{t1.strftime('%H:%M')} | {duration:.0f} min | "
                    f"score={filler['score']:.1f} | mean-rate={mean_rate:.2f}"
                )

                blocks.insert(best_index, filler)
                blocks = merge_touching(nid, blocks)
                added_windows += 1

            # No legal additional min-window block remains. Absorb residual idle time into adjacent windows.
            extension_steps = 0
            extension_by_source = {}
            i = 0
            while i < len(blocks) - 1:
                left = blocks[i]
                right = blocks[i + 1]
                reserve = 0 if left["source"] == right["source"] else overhead_steps

                while right["start_idx"] - left["end_idx"] > reserve:
                    li = left["end_idx"]
                    ri = right["start_idx"] - 1
                    left_can = li < right["start_idx"] - reserve and slot_valid[nid, left["source"]][li]
                    right_can = ri >= left["end_idx"] + reserve and slot_valid[nid, right["source"]][ri]

                    if not left_can and not right_can:
                        break

                    left_score = slot_scores[nid, left["source"]][li] if left_can else -np.inf
                    right_score = slot_scores[nid, right["source"]][ri] if right_can else -np.inf

                    if left_score >= right_score:
                        left["end_idx"] += 1
                        chosen = left["source"]
                    else:
                        right["start_idx"] -= 1
                        chosen = right["source"]

                    extension_steps += 1
                    extension_by_source[chosen] = extension_by_source.get(chosen, 0) + 1

                recalc_score(nid, left)
                recalc_score(nid, right)
                i += 1

            blocks = merge_touching(nid, blocks)

            # Independent invariants for this finalized night.
            if blocks[0]["start_idx"] != night_start or blocks[-1]["end_idx"] != night_end:
                raise RuntimeError(f"Night {nid + 1} no longer spans the full twilight-to-twilight interval.")

            for block in blocks:
                if not np.all(slot_valid[nid, block["source"]][block["start_idx"]:block["end_idx"]]):
                    raise RuntimeError(
                        f"Invalid visibility interval generated for {block['source']} on night {nid + 1}."
                    )

            for left, right in zip(blocks[:-1], blocks[1:]):
                required = 0 if left["source"] == right["source"] else overhead_steps
                if right["start_idx"] - left["end_idx"] < required:
                    raise RuntimeError(f"Overhead violation on night {nid + 1}.")

            on_target = sum(block["end_idx"] - block["start_idx"] for block in blocks)
            utilization = 100.0 * on_target / available if available else 0.0
            extension_text = ", ".join(
                f"{name} +{nstep * step_minutes:.0f}m"
                for name, nstep in sorted(extension_by_source.items(), key=lambda item: -item[1])
            )
            if not extension_text:
                extension_text = "none"

            print(
                f"[INFO] Stage 2 Night {nid + 1}: complete | fillers={added_windows} | "
                f"residual extension={extension_steps * step_minutes:.0f} min | "
                f"on-target={on_target * step_minutes:.0f} min ({utilization:.1f}%)"
            )
            print(f"[INFO] Stage 2 Night {nid + 1}: extension allocation: {extension_text}")
            return blocks

        used_stage2_nights = [nid for nid in night_ids if night_selected[nid]]
        print(
            f"[INFO] Stage 2: filling gaps on {len(used_stage2_nights)} selected night(s); "
            f"minimum extra window={min_window_minutes:.0f} min"
        )
        for nid in used_stage2_nights:
            night_selected[nid] = fill_night(nid, night_selected[nid])
        print("[INFO] Stage 2: gap filling and residual expansion complete.")

        # --------------------------------------------------------------------------------------------------
        # PHASE 3: globally slide the Stage-2 separator/overhead anchors.
        #
        # Stage 2 has already fixed the observing topology: selected nights, source order, number of windows,
        # and the idle/overhead width between every neighbouring pair. Stage 3 keeps that topology exactly and
        # jointly moves all internal separators. Because every separator width is fixed and both twilight edges
        # are fixed, total on-target time is invariant; no separate utilization optimization is required.
        #
        # Stage 3A: minimize the source-to-source spread in integrated visibility score.
        # Stage 3B: at that best balance, maximize the total integrated visibility score.
        # --------------------------------------------------------------------------------------------------
        def balance_schedule():
            used_nights = [nid for nid in night_ids if night_selected[nid]]
            if not used_nights or not selected_sources:
                return

            # ----------------------------------------------------------------------------------------------
            # Freeze the Stage-2 topology and find the contiguous visibility component containing each block.
            # Any moved block is constrained to remain inside this same component, which guarantees that every
            # slot in the moved interval remains visibility-valid without introducing source-by-slot binaries.
            # ----------------------------------------------------------------------------------------------
            topology = {}
            initial_visibility = {name: 0.0 for name in selected_sources}
            initial_exposure = {name: 0 for name in selected_sources}
            initial_on_target = 0

            for nid in used_nights:
                blocks = merge_touching(nid, night_selected[nid])
                blocks.sort(key=lambda block: block["start_idx"])
                night_selected[nid] = blocks
                night_start, night_end = night_bounds[nid]

                if blocks[0]["start_idx"] != night_start or blocks[-1]["end_idx"] != night_end:
                    raise RuntimeError(f"Stage-2 night {nid + 1} does not span twilight-to-twilight.")

                components = []
                for block in blocks:
                    name = block["source"]
                    start_idx = block["start_idx"]
                    end_idx = block["end_idx"]
                    valid = slot_valid[nid, name]

                    if end_idx - start_idx < min_window_steps:
                        raise RuntimeError(f"Stage-2 window shorter than min_window for {name}.")
                    if not np.all(valid[start_idx:end_idx]):
                        raise RuntimeError(f"Stage-2 window violates visibility for {name}.")

                    left_limit = start_idx
                    while left_limit > night_start and valid[left_limit - 1]:
                        left_limit -= 1

                    right_limit = end_idx
                    while right_limit < night_end and valid[right_limit]:
                        right_limit += 1

                    components.append((left_limit, right_limit))
                    initial_visibility[name] += float(np.sum(slot_scores[nid, name][start_idx:end_idx]))
                    initial_exposure[name] += end_idx - start_idx
                    initial_on_target += end_idx - start_idx

                separators = []
                for left, right in zip(blocks[:-1], blocks[1:]):
                    gap_steps = right["start_idx"] - left["end_idx"]
                    required = 0 if left["source"] == right["source"] else overhead_steps

                    if gap_steps < required:
                        raise RuntimeError(f"Stage-2 schedule violates overhead on night {nid + 1}.")

                    # Keep the complete Stage-2 separator width fixed. It is normally exactly the requested
                    # overhead. If visibility forced a larger idle gap, preserving that width guarantees that
                    # the Stage-2 schedule itself remains a feasible point of the anchor optimization.
                    separators.append(gap_steps)

                topology[nid] = {
                    "blocks": blocks,
                    "components": components,
                    "separators": separators,
                }

            if any(initial_exposure[name] < exptime_steps[name] for name in selected_sources):
                raise RuntimeError("Stage-2 schedule falls below a required source exposure before balancing.")

            # A one-window night has no movable separator. If every used night is like this, there is nothing
            # to rebalance and the Stage-2 schedule is already the unique topology-preserving solution.
            nanchors = sum(max(len(topology[nid]["blocks"]) - 1, 0) for nid in used_nights)
            if nanchors == 0:
                print("[INFO] Stage 3: no internal separator anchors to move; keeping the Stage-2 schedule.")
                return

            # ----------------------------------------------------------------------------------------------
            # One-hot anchor-position variables. Anchor (nid, i) is the END of block i. The following block
            # starts at anchor + fixed_separator_width. All anchors are optimized simultaneously, so moving an
            # early separator can propagate through every later window on that night.
            # ----------------------------------------------------------------------------------------------
            balance = pulp.LpProblem("anchor_balanced_schedule", pulp.LpMinimize)
            anchor_y = {}
            anchor_pos = {}

            for nid in used_nights:
                blocks = topology[nid]["blocks"]
                components = topology[nid]["components"]
                separators = topology[nid]["separators"]
                night_start, night_end = night_bounds[nid]

                for i, gap_steps in enumerate(separators):
                    left_component = components[i]
                    right_component = components[i + 1]

                    lo = max(
                        night_start,
                        left_component[0] + min_window_steps,
                        right_component[0] - gap_steps,
                    )
                    hi = min(
                        night_end - gap_steps,
                        left_component[1],
                        right_component[1] - gap_steps - min_window_steps,
                    )

                    current = blocks[i]["end_idx"]
                    if lo > hi or not lo <= current <= hi:
                        raise RuntimeError(
                            f"No feasible anchor domain around the Stage-2 separator {i + 1} "
                            f"on night {nid + 1}."
                        )

                    key = (nid, i)
                    positions = range(lo, hi + 1)
                    anchor_y[key] = {
                        t: pulp.LpVariable(f"anchor_{nid}_{i}_{t}", cat="Binary") for t in positions
                    }
                    balance += pulp.lpSum(anchor_y[key].values()) == 1
                    anchor_pos[key] = pulp.lpSum(t * anchor_y[key][t] for t in positions)

            # ----------------------------------------------------------------------------------------------
            # Block starts/ends/durations are affine functions of adjacent anchor positions. Constraining them
            # to their original contiguous visibility component guarantees all moved windows remain valid.
            # ----------------------------------------------------------------------------------------------
            block_geometry = {}

            for nid in used_nights:
                blocks = topology[nid]["blocks"]
                components = topology[nid]["components"]
                separators = topology[nid]["separators"]
                night_start, night_end = night_bounds[nid]

                for j, block in enumerate(blocks):
                    start_expr = night_start if j == 0 else anchor_pos[nid, j - 1] + separators[j - 1]
                    end_expr = night_end if j == len(blocks) - 1 else anchor_pos[nid, j]
                    duration_expr = end_expr - start_expr
                    left_limit, right_limit = components[j]

                    balance += start_expr >= left_limit
                    balance += end_expr <= right_limit
                    balance += duration_expr >= min_window_steps
                    block_geometry[nid, j] = (start_expr, end_expr, duration_expr)

            # ----------------------------------------------------------------------------------------------
            # Required total exposure per source. A source may have several windows and/or several nights;
            # only its summed exposure must remain above its requested exptime.
            # ----------------------------------------------------------------------------------------------
            exposure = {}
            for name in selected_sources:
                durations = [
                    block_geometry[nid, j][2]
                    for nid in used_nights
                    for j, block in enumerate(topology[nid]["blocks"])
                    if block["source"] == name
                ]
                exposure[name] = pulp.lpSum(durations)
                balance += exposure[name] >= exptime_steps[name]

            # ----------------------------------------------------------------------------------------------
            # Integrated visibility is linearized with prefix sums and the one-hot anchor choices. For a block
            # [a, b), score = P(b) - P(a). Fixed twilight edges contribute constants; movable edges contribute
            # lookup values selected by the corresponding anchor-position binary.
            # ----------------------------------------------------------------------------------------------
            prefix_score = {}
            for nid in used_nights:
                for name in selected_sources:
                    values = np.asarray(slot_scores[nid, name], dtype=float)
                    prefix_score[nid, name] = np.concatenate(
                        ([0.0], np.cumsum(np.where(np.isfinite(values), values, 0.0)))
                    )

            source_terms = {name: [] for name in selected_sources}

            for nid in used_nights:
                blocks = topology[nid]["blocks"]
                separators = topology[nid]["separators"]
                night_start, night_end = night_bounds[nid]

                for j, block in enumerate(blocks):
                    name = block["source"]
                    prefix = prefix_score[nid, name]
                    score_expr = 0

                    if j == 0:
                        score_expr -= float(prefix[night_start])
                    else:
                        key = (nid, j - 1)
                        gap_steps = separators[j - 1]
                        score_expr -= pulp.lpSum(
                            float(prefix[t + gap_steps]) * anchor_y[key][t]
                            for t in anchor_y[key]
                        )

                    if j == len(blocks) - 1:
                        score_expr += float(prefix[night_end])
                    else:
                        key = (nid, j)
                        score_expr += pulp.lpSum(
                            float(prefix[t]) * anchor_y[key][t] for t in anchor_y[key]
                        )

                    source_terms[name].append(score_expr)

            source_visibility = {
                name: pulp.lpSum(source_terms[name]) for name in selected_sources
            }
            total_visibility = pulp.lpSum(source_visibility.values())

            min_visibility = pulp.LpVariable("min_source_visibility", lowBound=None)
            max_visibility = pulp.LpVariable("max_source_visibility", lowBound=None)
            for name in selected_sources:
                balance += min_visibility <= source_visibility[name]
                balance += max_visibility >= source_visibility[name]

            visibility_spread = max_visibility - min_visibility
            balance_solver = pulp.HiGHS(msg=False, threads=nthreads, parallel="on", gapRel=0.0)
            nplacement_vars = sum(len(choices) for choices in anchor_y.values())

            initial_spread = max(initial_visibility.values()) - min(initial_visibility.values())
            print(
                f"[INFO] Stage 3: jointly optimizing {nanchors} separator anchor(s) "
                f"with {nplacement_vars:,} placement variables"
            )
            print(f"[INFO] Stage 3A: minimizing visibility spread from the Stage-2 value {initial_spread:.2f} …")

            # ----------------------------------------------------------------------------------------------
            # Stage 3A: globally equalize integrated visibility scores.
            # ----------------------------------------------------------------------------------------------
            balance.setObjective(visibility_spread)
            balance.solve(balance_solver)

            status = pulp.LpStatus[balance.status]
            if status not in ("Optimal", "Feasible"):
                raise RuntimeError(f"Stage 3A failed with status: {status}")

            best_spread = float(pulp.value(visibility_spread) or 0.0)
            spread_eps = max(1e-6, abs(best_spread) * 1e-8)
            balance += visibility_spread <= best_spread + spread_eps

            # ----------------------------------------------------------------------------------------------
            # Stage 3B: among schedules with the best balance, maximize the total integrated visibility.
            # ----------------------------------------------------------------------------------------------
            print("[INFO] Stage 3B: maximizing total visibility at the best anchor balance …")
            balance.sense = pulp.LpMaximize
            balance.setObjective(total_visibility)
            balance.solve(balance_solver)

            status = pulp.LpStatus[balance.status]
            if status not in ("Optimal", "Feasible"):
                raise RuntimeError(f"Stage 3B failed with status: {status}")

            # ----------------------------------------------------------------------------------------------
            # Reconstruct the moved windows from the jointly optimized anchor positions.
            # ----------------------------------------------------------------------------------------------
            anchor_values = {
                key: int(round(pulp.value(expr))) for key, expr in anchor_pos.items()
            }
            balanced = {nid: [] for nid in night_ids}

            for nid in used_nights:
                blocks = topology[nid]["blocks"]
                separators = topology[nid]["separators"]
                night_start, night_end = night_bounds[nid]

                for j, old_block in enumerate(blocks):
                    start_idx = night_start if j == 0 else anchor_values[nid, j - 1] + separators[j - 1]
                    end_idx = night_end if j == len(blocks) - 1 else anchor_values[nid, j]
                    name = old_block["source"]

                    balanced[nid].append({
                        "source": name,
                        "src_obj": source_map[name],
                        "night_id": nid,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "score": float(np.sum(slot_scores[nid, name][start_idx:end_idx])),
                        "core": old_block.get("core", False),
                    })

            # ----------------------------------------------------------------------------------------------
            # Independent invariants. These checks do not use the optimization expressions.
            # ----------------------------------------------------------------------------------------------
            final_visibility = {name: 0.0 for name in selected_sources}
            final_exposure = {name: 0 for name in selected_sources}
            final_on_target = 0

            for nid in used_nights:
                blocks = balanced[nid]
                old_blocks = topology[nid]["blocks"]
                components = topology[nid]["components"]
                separators = topology[nid]["separators"]
                night_start, night_end = night_bounds[nid]

                if len(blocks) != len(old_blocks):
                    raise RuntimeError(f"Stage 3 changed the observing topology on night {nid + 1}.")
                if blocks[0]["start_idx"] != night_start or blocks[-1]["end_idx"] != night_end:
                    raise RuntimeError(f"Balanced night {nid + 1} does not span twilight-to-twilight.")

                for j, block in enumerate(blocks):
                    name = block["source"]
                    start_idx = block["start_idx"]
                    end_idx = block["end_idx"]
                    left_limit, right_limit = components[j]

                    if name != old_blocks[j]["source"]:
                        raise RuntimeError(f"Stage 3 changed the source order on night {nid + 1}.")
                    if end_idx - start_idx < min_window_steps:
                        raise RuntimeError(f"Balanced window shorter than min_window for {name}.")
                    if start_idx < left_limit or end_idx > right_limit:
                        raise RuntimeError(f"Balanced window left its allowed visibility component for {name}.")
                    if not np.all(slot_valid[nid, name][start_idx:end_idx]):
                        raise RuntimeError(f"Balanced window violates visibility for {name}.")

                    final_exposure[name] += end_idx - start_idx
                    final_visibility[name] += float(np.sum(slot_scores[nid, name][start_idx:end_idx]))
                    final_on_target += end_idx - start_idx

                for i, (left, right) in enumerate(zip(blocks[:-1], blocks[1:])):
                    actual_gap = right["start_idx"] - left["end_idx"]
                    if actual_gap != separators[i]:
                        raise RuntimeError(f"Stage 3 changed separator width on night {nid + 1}.")

                    required = 0 if left["source"] == right["source"] else overhead_steps
                    if actual_gap < required:
                        raise RuntimeError(f"Balanced schedule violates overhead on night {nid + 1}.")

            if final_on_target != initial_on_target:
                raise RuntimeError("Stage 3 changed total telescope utilization despite fixed separators.")

            for name in selected_sources:
                if final_exposure[name] < exptime_steps[name]:
                    raise RuntimeError(f"Balanced schedule falls below required exptime for {name}.")

            final_spread = max(final_visibility.values()) - min(final_visibility.values())
            print(
                f"[INFO] Stage 3: visibility spread {initial_spread:.2f} -> {final_spread:.2f} | "
                f"on-target time unchanged at {final_on_target * step_minutes:.0f} min"
            )

            for name in sorted(selected_sources):
                mean_rate = final_visibility[name] / max(final_exposure[name], 1)
                print(
                    f"       {name:<24s} exposure={final_exposure[name] * step_minutes:5.0f} min | "
                    f"visibility={final_visibility[name]:8.2f} | mean-rate={mean_rate:7.2f}"
                )

            for nid in used_nights:
                night_selected[nid] = balanced[nid]

        balance_schedule()

        # --------------------------------------------------------------------------------------------------
        # Final sequences
        # --------------------------------------------------------------------------------------------------
        sequences = []

        for nid in night_ids:
            blocks = night_selected[nid]
            if not blocks:
                continue

            blocks.sort(key=lambda block: block["start_idx"])
            times = self.__nightly_data[nid]["astropy_times"]
            sequence_blocks = [
                ScheduledBlock(
                    source=block["src_obj"],
                    start_time=times[block["start_idx"]],
                    end_time=times[block["end_idx"]],
                    score=block["score"],
                )
                for block in blocks
            ]
            sequences.append({"night_id": nid, "sequence_blocks": sequence_blocks})

        self.__observation_sequences = sequences

        print(
            f"[INFO] Final schedule: {sum(len(seq['sequence_blocks']) for seq in sequences)} windows | "
            f"{len(selected_sources)} completed sources | {len(sequences)} night(s)"
        )

        for seq in sequences:
            print(f"[INFO] Night {seq['night_id'] + 1}:")
            for block in seq["sequence_blocks"]:
                t0_utc = block.start_time.datetime.replace(tzinfo=timezone.utc)
                t1_utc = block.end_time.datetime.replace(tzinfo=timezone.utc)
                t0_local = self.obs.utc_to_local(t0_utc)
                t1_local = self.obs.utc_to_local(t1_utc)
                duration = (t1_utc - t0_utc).total_seconds() / 60.0
                print(
                    f"       {block.source.name:<24s} {t0_local.strftime('%H:%M')}–"
                    f"{t1_local.strftime('%H:%M')} ({duration:.0f} min)"
                )

    # -----------------------------------------------------------------------------
    # Stand-alone plotting function
    # -----------------------------------------------------------------------------

    def plot_tracking(
        self, min_altitude: float, save_path: str, second_axis="lst"
    ) -> None:
        """
        Plot altitude vs time with dual x-axes (local clock time + LST) and a
        smooth day/night background gradient.

        Parameters
        ----------
        all_nights_data : list[dict] — one night-data dict per night.
        observatory     : Observatory instance.
        obs_sequences   : List of observation sequences, one per night.
        save_path       : Output path (.pdf).
        second_axis     : The secondary x-axis to display ('lst', 'ut1', 'utc').
        """
        if not HAS_MPL:
            print("[ERROR] matplotlib is required.  pip install matplotlib")
            return

        obs_sequences = self.__observation_sequences
        if not obs_sequences:
            print("[WARN] No observation sequences to plot.")
            return
        save_path = save_path.strip() + ".pdf"
        print(f"[INFO] Plotting optimized plan ({len(obs_sequences)} night(s)) → {save_path} …")
        # -------------------------------------------------------------------------
        def generate_palette(n):
            colors = []
            for i in range(n):
                h = i / n              # evenly spaced in [0,1)
                s = 1.0                # max saturation
                v = 1.0                # max brightness
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                colors.append('#{:02X}{:02X}{:02X}'.format(
                    int(r * 255),
                    int(g * 255),
                    int(b * 255)
                ))
            return colors
        # -------------------------------------------------------------------------
        def _sun_alt_to_sky_color(
            sun_alt_deg: float,
            moon_alt_deg: float = -90.0,
            moon_illumination: float = 0.0,
        ) -> tuple[float, float, float]:
            """
            Map Sun altitude + Moon state to an RGB sky colour.
            Moon above horizon adds a grey-blue brightening scaled by illumination.
            """
            # --- Base sky from sun altitude ---
            stops = [
                ( 0.0, (0.529, 0.808, 0.922)),  # full day
                (-6.0, (0.95,  0.55,  0.25 )),  # civil twilight
                (-12.0,(0.15,  0.12,  0.35 )),  # nautical twilight
                (-18.0,(0.043, 0.055, 0.102)),  # astronomical night
            ]

            if sun_alt_deg >= stops[0][0]:
                base = stops[0][1]
            elif sun_alt_deg <= stops[-1][0]:
                base = stops[-1][1]
            else:
                base = stops[-1][1]
                for i in range(len(stops) - 1):
                    alt_hi, rgb_hi = stops[i]
                    alt_lo, rgb_lo = stops[i + 1]
                    if alt_lo <= sun_alt_deg <= alt_hi:
                        t = (alt_hi - sun_alt_deg) / (alt_hi - alt_lo)
                        t = (1.0 - math.cos(math.pi * t)) / 2.0
                        base = (
                            rgb_hi[0] * (1 - t) + rgb_lo[0] * t,
                            rgb_hi[1] * (1 - t) + rgb_lo[1] * t,
                            rgb_hi[2] * (1 - t) + rgb_lo[2] * t,
                        )
                        break

            # --- Moon contribution ---
            # Only applies when moon is above horizon and sun is below -6
            # (during full day/civil twilight the sun dominates, moon is invisible)
            if moon_alt_deg > 0.0 and sun_alt_deg < -6.0:
                # altitude factor: ramps 0→1 over [0°, 45°], clamped above
                alt_factor = min(moon_alt_deg / 45.0, 1.0)
                alt_factor = (1.0 - math.cos(math.pi * alt_factor)) / 2.0

                # brighter, more visible cool white-blue
                moon_rgb = (0.55, 0.60, 0.75)

                # strength: illumination drives it hard, altitude modulates
                # cap raised to 0.65 so full moon high in sky is clearly visible
                strength = alt_factor * (moon_illumination ** 0.5) * 0.65

                r = base[0] * (1 - strength) + moon_rgb[0] * strength
                g = base[1] * (1 - strength) + moon_rgb[1] * strength
                b = base[2] * (1 - strength) + moon_rgb[2] * strength
                return (r, g, b)

            return base
        # -------------------------------------------------------------------------
        def _build_sky_background(
            local_times: np.ndarray,
            sun_alts_arr: np.ndarray,
            moon_alts_arr: np.ndarray,
            illum_arr: np.ndarray,
            ymin: float,
            ymax: float,
        ) -> mpatches.PatchCollection:
            """
            Build a PatchCollection of sky background rectangles blending sun and moon
            contributions. Add directly to the axis with ax.add_collection(patch).
            """
            rects  = []
            colors = []

            for i in range(len(local_times) - 1):
                col0 = _sun_alt_to_sky_color(
                    sun_alts_arr[i],
                    moon_alt_deg=float(moon_alts_arr[i]),
                    moon_illumination=float(illum_arr[i]),
                )
                col1 = _sun_alt_to_sky_color(
                    sun_alts_arr[i + 1],
                    moon_alt_deg=float(moon_alts_arr[i + 1]),
                    moon_illumination=float(illum_arr[i + 1]),
                )
                col = tuple((a + b) / 2 for a, b in zip(col0, col1))

                x0 = mdates.date2num(local_times[i])
                x1 = mdates.date2num(local_times[i + 1])
                rects.append(mpatches.Rectangle((x0, ymin), x1 - x0, ymax - ymin))
                colors.append(col)

            collection = PatchCollection(
                rects,
                facecolors=colors,
                edgecolors="none",
                zorder=0,
            )
            return collection
        
        # -------------------------------------------------------------------------
        def _draw_night(night_data: dict, obs_seq: list[ScheduledBlock]) -> plt.Figure:
            """Render one page onto a new Figure and return it."""
            astropy_times = night_data["astropy_times"]
            local_times   = np.array([self.obs.utc_to_local(t.datetime) for t in astropy_times])
            src_altazs    = night_data["src_altazs"]   
            moon_altazs   = night_data["moon"]         
            sun_altazs    = night_data["sun"]       
            sequence_blocks = {}
            for block in obs_seq:
                sequence_blocks.setdefault(block.source.name, []).append(block)

            # Extract scalar float arrays for plotting
            moon_alts_arr = moon_altazs.alt.deg
            sun_alts_arr  = sun_altazs.alt.deg

            date_str = (
                f"{local_times[0].strftime('%Y-%m-%d %H:%M')} \u2192 "
                f"{local_times[-1].strftime('%Y-%m-%d %H:%M')} {self.obs.timezone}"
            )

            fig, ax_local = plt.subplots(figsize=(14, 8))

            # -- y limits (set before drawing background) -------------------------
            ymin, ymax = max(min(min_altitude, 15), 0), 90
            ax_local.set_ylim(ymin, ymax)
            ax_local.set_xlim(local_times[0], local_times[-1])

            _, illum_arr = compute_moon_illumination(sun_altazs, moon_altazs)
            ax_local.add_collection(_build_sky_background(
                local_times    = local_times,
                sun_alts_arr   = sun_alts_arr,
                moon_alts_arr  = moon_alts_arr,
                illum_arr      = illum_arr,
                ymin           = ymin,
                ymax           = ymax,
            ))

            # Choose text/line colours that contrast with the background
            fg_color      = "#1a1a2e"
            fg_dim        = "#555577"
            grid_color    = "#c0c8d8"
            legend_face   = "#f0f4ff"
            legend_edge   = "#9090b0"
            spine_color   = "#9090b0"
            moon_col      = "#F7ECB0"
            minalt_color  = "#87A2BE"
            ref_color     = "#8899aa"
            lst_color     = "#334455"

            # -- Source altitude tracks -------------------------------------------
            palette = generate_palette(len(src_altazs))
            
            for idx, (name, src_altaz_arr) in enumerate(src_altazs.items()):
                color    = palette[idx % len(palette)]
                alts_arr = np.array(src_altaz_arr.alt.deg)

                # Faint dashed track for the full arc
                ax_local.plot(local_times, alts_arr, color=color, linewidth=1.0, 
                            linestyle="--", alpha=0.4, zorder=2)

                if name in sequence_blocks:
                    for j, seq_block in enumerate(sequence_blocks[name]):
                        selected_times = (astropy_times >= seq_block.start_time) & (astropy_times <= seq_block.end_time)
                        local_times_sel = local_times[selected_times]
                        alts_sel = alts_arr[selected_times]
                        ax_local.plot(
                            local_times_sel, alts_sel, color=color, linewidth=3, linestyle="solid",
                            alpha=0.8, zorder=3, label=name if j == 0 else None,
                        )

            # -- Moon track -------------------------------------------------------
            if np.any(moon_alts_arr > 0):
                ax_local.plot(local_times, moon_alts_arr, color=moon_col, label="Moon",
                              linewidth=2, linestyle=(5, (10, 3)), alpha=0.6, zorder=2)

            # # -- Reference lines --------------------------------------------------
            ax_local.axhline(
                min_altitude, color=minalt_color, linewidth=2, linestyle="solid", zorder=1, label="Min Altitude"
            )
            # ax_local.axhline(60, color=ref_color, linewidth=0.5, linestyle=":", zorder=1)

            # -- Axes styling -----------------------------------------------------
            ax_local.set_ylabel("Altitude (deg)", color=fg_color, fontsize=12)
            ax_local.yaxis.set_major_locator(mticker.MultipleLocator(10))
            ax_local.tick_params(axis="y", colors=fg_color)
            ax_local.yaxis.grid(True, color=grid_color, linewidth=0.1, zorder=0)
            ax_local.set_axisbelow(True)

            ax_local.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_local.xaxis.set_major_locator(mdates.HourLocator())
            plt.setp(ax_local.xaxis.get_majorticklabels(), rotation=0, color=fg_color, fontsize=9)

            # Use the midpoint UTC time for the offset label so DST is correct
            mid_time  = astropy_times[len(astropy_times) // 2]
            tz_label = self.obs.tz_label_at(mid_time.datetime)
            ax_local.set_xlabel(f"Local Time  ({tz_label}  {self.obs.timezone})", color=fg_color, fontsize=11)

            # ------------------------------------------------------------------
            # plot airmass on a twin y-axis, using the same y-ticks as altitude
            ax_airmass = ax_local.twinx()
            ax_airmass.set_ylim(ymin, ymax)
            alt_ticks = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
            airmass_ticks = compute_airmass(alt_ticks)
            ax_airmass.set_yticks(alt_ticks)
            ax_airmass.set_yticklabels([f"{am:.1f}" if am<10 else "" for am in airmass_ticks], color=fg_dim, fontsize=9)
            ax_airmass.set_ylabel("Airmass", color=fg_dim, fontsize=10)
            ax_airmass.tick_params(axis="y", colors=fg_dim)

            # -- Twin x-axis for (UT1 / LST / UTC) -----------------------------
            ax2 = ax_local.twiny()
            ax2.set_xlim(ax_local.get_xlim())
            # Use same tick positions as primary axis
            xticks = ax_local.get_xticks()
            ax2.set_xticks(xticks)
            utc_ticks = [self.obs.local_to_utc(t.replace(tzinfo=None)) for t in mdates.num2date(xticks, tz=None)]
            astropy_ticks = Time(utc_ticks, scale="utc")
            if second_axis == "ut1":
                ut1_dt = astropy_ticks.ut1.to_datetime()
                labels = [t.strftime("%H:%M") for t in ut1_dt]
                ax2.set_xticklabels(labels, color=lst_color, fontsize=9, rotation=0)
                ax2.set_xlabel("UT1", color=lst_color, fontsize=11)
            elif second_axis == "utc":
                utc_dt = astropy_ticks.to_datetime()
                labels = [t.strftime("%H:%M") for t in utc_dt]
                ax2.set_xticklabels(labels, color=lst_color, fontsize=9, rotation=0)
                ax2.set_xlabel("UTC", color=lst_color, fontsize=11)
            elif second_axis == "lst":
                lon = self.obs.longitude * au.deg
                lst = astropy_ticks.sidereal_time("apparent", longitude=lon)
                labels = []
                for h in lst.hour:
                    h_mod = h % 24
                    hh = int(h_mod)
                    mm = int(round((h_mod - hh) * 60)) % 60
                    labels.append(f"{hh:02d}:{mm:02d}")
                ax2.set_xticklabels(labels, color=lst_color, fontsize=9, rotation=0)
                ax2.set_xlabel("Local Sidereal Time", color=lst_color, fontsize=11)
            ax2.tick_params(axis="x", colors=lst_color)

            for spine in ax2.spines.values():
                spine.set_edgecolor(spine_color)
            for spine in ax_local.spines.values():
                spine.set_edgecolor(spine_color)

            # Figure / axes backgrounds — transparent so gradient shows
            fig.patch.set_facecolor("none")
            ax_local.set_facecolor("none")

            ax_local.set_title(
                f"ZENITHS Night Chart — {self.obs.name}  "
                f"(lat {self.obs.latitude:+.2f}°, lon {self.obs.longitude:+.2f}°)"
                f" — {date_str}",
                color=fg_color, fontsize=12, pad=14,
            )
            ax_local.legend(
                facecolor=legend_face, edgecolor=legend_edge,
                labelcolor=fg_color, fontsize=9, framealpha=0.88,
                loc='lower center', ncols=3
            )

            plt.subplots_adjust(right=0.92, top=0.92, bottom=0.08, left=0.08)
            return fig
        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------

        with PdfPages(save_path) as pdf:
            metadata = pdf.infodict()
            metadata["Title"] = f"{APP_NAME} Observation Schedule — {self.obs.name}"
            metadata["Subject"] = APP_EXPANSION
            metadata["Creator"] = f"{APP_NAME} {__version__}"
            metadata["Author"] = "ZENITHS"

            for obs_seq in obs_sequences:
                night_data = self.__nightly_data[obs_seq['night_id']]
                fig = _draw_night(night_data, obs_seq['sequence_blocks'])
                pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=300)
                plt.close(fig)

            print(f"[INFO] PDF saved to {save_path!r}.")

    def write_observation_report(
        self, start_utc: datetime, end_utc: datetime, min_altitude: float,
        moon_exclusion_deg: float, max_airmass: float, exptime_minutes: float,
        min_window_minutes: float, overhead_minutes: float, step_minutes: float,
        sun_alt_limit: float, output_path: str,
    ) -> None:
        """
        Write the ZENITHS observation plan as a TSV file with # comment header lines.
        """

        scheduled_names = {
            block.source.name
            for seq in self.__observation_sequences
            for block in seq["sequence_blocks"]
        }
        unscheduled = [s for s in self.sources if s.name not in scheduled_names]
        lines = []

        # -------------------------------------------------------------------------
        # Comment header — all prefixed with #
        # -------------------------------------------------------------------------
        lines.append("# ============================================================")
        lines.append("# ZENITHS OBSERVATION SCHEDULE")
        lines.append(f"# {APP_EXPANSION}")
        lines.append(f"# Version           : {__version__}")
        lines.append(f"# Generated         : {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("# ------------------------------------------------------------")
        lines.append("# OBSERVATORY")
        lines.append(f"# Name              : {self.obs.name}")
        lines.append(f"# Latitude          : {self.obs.latitude:+.4f} deg")
        lines.append(f"# Longitude         : {self.obs.longitude:+.4f} deg")
        lines.append(f"# Elevation         : {self.obs.elevation:.0f} m")
        lines.append(f"# Timezone          : {self.obs.timezone}")
        lines.append("# ------------------------------------------------------------")
        lines.append("# REQUEST WINDOW")
        lines.append(f"# Start (UTC)       : {start_utc.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"# End   (UTC)       : {end_utc.strftime('%Y-%m-%d %H:%M')}")
        lines.append(
            f"# Start (local)     : {self.obs.utc_to_local(start_utc).strftime('%Y-%m-%d %H:%M')}  "
            f"{self.obs.tz_label_at(start_utc)}"
        )
        lines.append(
            f"# End   (local)     : {self.obs.utc_to_local(end_utc).strftime('%Y-%m-%d %H:%M')}  "
            f"{self.obs.tz_label_at(end_utc)}"
        )
        lines.append("# ------------------------------------------------------------")
        lines.append("# CONSTRAINTS")
        lines.append(f"# Min altitude      : {min_altitude:.1f} deg")
        lines.append(f"# Moon exclusion    : {moon_exclusion_deg:.1f} deg")
        lines.append(f"# Max airmass       : {max_airmass:.2f}")
        lines.append(f"# Default exptime   : {exptime_minutes:.0f} min")
        lines.append(f"# Min window        : {min_window_minutes:.0f} min")
        lines.append(f"# Slew overhead     : {overhead_minutes:.0f} min")
        lines.append(f"# Sampling cadence  : {step_minutes:.1f} min")
        lines.append(f"# Twilight limit    : Sun <= {sun_alt_limit:.1f} deg")
        lines.append(f"# Bright time       : {'yes' if self.bright_time else 'no'}")
        lines.append("# ------------------------------------------------------------")
        lines.append("# VISIBILITY SCORE")
        lines.append("# Meaning           : relative observing-efficiency proxy; higher = more useful time")
        lines.append("# Formula           : 100 x brightness_eff x airmass_eff x sun_eff x moon_eff")
        if self.mag_reference is None:
            lines.append("# Magnitude term    : disabled (no finite magnitudes supplied)")
        else:
            lines.append(f"# Magnitude ref     : sample median = {self.mag_reference:.3f} mag")
            lines.append("# Magnitude scaling : 10^[-0.4 (mag - reference)], clipped to [0.10, 10]")
        lines.append(f"# Extinction coeff  : {self.extinction_mag_per_airmass:.3f} mag/airmass")
        lines.append(
            f"# Twilight soft     : efficiency {self.sun_edge_efficiency:.2f} at the hard limit, "
            f"1.00 after {self.sun_soft_range_deg:.1f} deg deeper darkness"
        )
        lines.append(f"# Moon score term   : {'disabled' if self.bright_time else 'enabled'}")
        lines.append("# ------------------------------------------------------------")
        lines.append(f"# SUMMARY")
        lines.append(f"# Nights scheduled  : {len(self.__observation_sequences)}")
        lines.append(f"# Sources scheduled : {len(scheduled_names)}")
        lines.append(f"# Sources total     : {len(self.sources)}")
        lines.append(f"# Unscheduled       : {len(unscheduled)}")
        if unscheduled:
            for src in unscheduled:
                lines.append(f"#   - {src.name}  RA {src.ra_hms}  Dec {src.dec_dms}")
        lines.append("# ------------------------------------------------------------")
        lines.append("# NIGHT SUMMARIES")
        for seq_idx, seq in enumerate(self.__observation_sequences, 1):
            nid        = seq["night_id"]
            night_data = self.__nightly_data[nid]
            blocks     = seq["sequence_blocks"]
            astropy_times = night_data["astropy_times"]
            sun_altazs    = night_data["sun"]
            moon_altazs   = night_data["moon"]
            sun_alts = sun_altazs.alt.deg
            night_mask = sun_alts <= sun_alt_limit
            night_indices = np.flatnonzero(night_mask)

            night_start_utc = astropy_times[night_indices[0]].datetime.replace(tzinfo=timezone.utc)
            night_end_utc = astropy_times[night_indices[-1]].datetime.replace(tzinfo=timezone.utc)

            total_min = sum(
                (b.end_time.datetime - b.start_time.datetime).total_seconds() / 60.0
                for b in blocks
            )
            tz_label = self.obs.tz_label_at(night_start_utc)
            lines.append(f"#   Night {seq_idx}: {night_start_utc.strftime('%Y-%m-%d')}")
            lines.append(
                f"#     Window (UTC)   : {night_start_utc.strftime('%Y-%m-%d %H:%M')} -> "
                f"{night_end_utc.strftime('%Y-%m-%d %H:%M')}"
            )
            lines.append(
                f"#     Window (local) : {self.obs.utc_to_local(night_start_utc).strftime('%Y-%m-%d %H:%M')} -> "
                f"{self.obs.utc_to_local(night_end_utc).strftime('%Y-%m-%d %H:%M')}  {tz_label}"
            )
            lines.append(f"#     Blocks         : {len(blocks)}  |  Total scheduled : {total_min:.1f} min")
        lines.append("# ============================================================")
        lines.append("")

        # -------------------------------------------------------------------------
        # TSV column header
        # -------------------------------------------------------------------------
        columns = [
            "source_name",
            "ra_deg",
            "dec_deg",
            "magnitude",
            "exptime",
            "night_date",
            "dark_start_local",
            "dark_end_local",
            "obs_start_local",
            "obs_end_local",
            "duration_min",
            "is_moon_up",
            "max_moon_illum_frac",
            "min_moon_sep_deg",
            "max_airmass",
            "min_altitude_deg",
            "score",
            "mean_score_rate",
        ]
        lines.append("\t".join(columns))

        # -------------------------------------------------------------------------
        # Data rows
        # -------------------------------------------------------------------------
        for seq in self.__observation_sequences:
            nid           = seq["night_id"]
            night_data    = self.__nightly_data[nid]
            blocks        = seq["sequence_blocks"]
            astropy_times = night_data["astropy_times"]
            src_altazs    = night_data["src_altazs"]
            sun_altazs    = night_data["sun"]
            moon_altazs   = night_data["moon"]

            phase_angle, illum = compute_moon_illumination(sun_altazs, moon_altazs)

            # True dark window — strip padding by finding where sun < -18
            sun_alts   = sun_altazs.alt.deg
            dark_mask = sun_alts <= sun_alt_limit
            dark_indices = np.where(dark_mask)[0]
            if len(dark_indices):
                dark_start_utc = astropy_times[dark_indices[0]].datetime.replace(tzinfo=timezone.utc)
                dark_end_utc   = astropy_times[dark_indices[-1]].datetime.replace(tzinfo=timezone.utc)
            else:
                dark_start_utc = astropy_times[0].datetime.replace(tzinfo=timezone.utc)
                dark_end_utc   = astropy_times[-1].datetime.replace(tzinfo=timezone.utc)

            dark_start_loc = self.obs.utc_to_local(dark_start_utc)
            dark_end_loc   = self.obs.utc_to_local(dark_end_utc)
            night_date     = dark_start_loc.strftime("%Y-%m-%d")
            tz_label       = self.obs.tz_label_at(dark_start_utc)

            for block in blocks:
                src    = block.source
                t0     = block.start_time
                t1     = block.end_time

                t0_utc = t0.datetime.replace(tzinfo=timezone.utc)
                t1_utc = t1.datetime.replace(tzinfo=timezone.utc)
                t0_loc = self.obs.utc_to_local(t0_utc)
                t1_loc = self.obs.utc_to_local(t1_utc)
                dur_min = (t1_utc - t0_utc).total_seconds() / 60.0

                src_altaz  = src_altazs[src.name]
                block_mask = (astropy_times >= t0) & (astropy_times <= t1)
                indices    = np.where(block_mask)[0]

                # stats over the full block window
                block_alts     = src_altaz.alt.deg[indices]
                block_ams      = compute_airmass(block_alts)
                block_moon_alt = moon_altazs.alt.deg[indices]
                block_moon_sep = np.array([
                    float(src_altaz[i].separation(moon_altazs[i]).deg)
                    for i in indices
                ])
                block_illum    = illum[indices]

                is_moon_up       = bool(np.any(block_moon_alt > 0.0))
                max_illum        = float(np.max(block_illum))
                min_moon_sep     = float(np.min(block_moon_sep))
                max_airmass_val  = float(np.max(block_ams[block_ams < 99.0]))
                min_alt          = float(np.min(block_alts))

                row = [
                    src.name,
                    f"{src.ra_deg:.6f}",
                    f"{src.dec_deg:.6f}",
                    "" if src.magnitude is None else f"{src.magnitude:.2f}",
                    "" if src.exptime is None else f"{src.exptime:.2f}",
                    night_date,
                    dark_start_loc.strftime("%Y-%m-%d %H:%M"),
                    dark_end_loc.strftime("%Y-%m-%d %H:%M"),
                    t0_loc.strftime("%Y-%m-%d %H:%M"),
                    t1_loc.strftime("%Y-%m-%d %H:%M"),
                    f"{dur_min:.1f}",
                    "1" if is_moon_up else "0",
                    f"{max_illum:.3f}",
                    f"{min_moon_sep:.3f}",
                    f"{max_airmass_val:.4f}",
                    f"{min_alt:.3f}",
                    f"{block.score:.3f}",
                    f"{block.score / max(dur_min / step_minutes, 1.0):.3f}",
                ]
                lines.append("\t".join(row))

        lines.append("")

        with open(output_path + ".tsv", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        print(f"[INFO] ZENITHS observation report written to {output_path + '.tsv'!r}.")

# -----------------------------------------------------------------------------
# Text-file source loader
# -----------------------------------------------------------------------------

def load_sources_from_file(filepath: str, default_exptime: float) -> list[SkySource]:
    """
    Load a whitespace- or comma-separated source table.

    Columns
    -------
    name, ra, dec, magnitude, exptime

    RA and Dec must be decimal degrees. A header row is optional. ``ra_deg``/``dec_deg`` and ``mag`` are
    accepted header aliases. Missing magnitude is allowed; missing exptime falls back to ``--exptime``.
    """

    cols = ["name", "ra", "dec", "magnitude", "exptime"]
    aliases = {"ra_deg": "ra", "dec_deg": "dec", "mag": "magnitude", "exp": "exptime"}
    missing = {"-": None, "N/A": None, "n/a": None, "?": None, "": None}

    first_fields = None
    with open(filepath, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            first_fields = [field.strip().lower() for field in re.split(r"[\s,]+", line)]
            break

    if first_fields is None:
        print(f"[INFO] Loaded 0 source(s) from {filepath!r}.")
        return []

    normalized_first = [aliases.get(field, field) for field in first_fields]
    has_header = len(normalized_first) >= 3 and normalized_first[:3] == ["name", "ra", "dec"]

    if has_header:
        df = pd.read_csv(
            filepath, sep=r"[\s,]+", comment="#", header=0, engine="python", dtype=str, on_bad_lines="warn",
        )
        df.columns = [aliases.get(str(col).strip().lower(), str(col).strip().lower()) for col in df.columns]
        missing_columns = [col for col in ("name", "ra", "dec") if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required source-table column(s): {', '.join(missing_columns)}")
        for col in ("magnitude", "exptime"):
            if col not in df.columns:
                df[col] = None
        df = df[cols]
    else:
        df = pd.read_csv(
            filepath, sep=r"[\s,]+", comment="#", header=None, engine="python",
            names=cols, dtype=str, on_bad_lines="warn",
        )

    df = df.dropna(how="all").reset_index(drop=True)
    for col in ("ra", "dec", "magnitude", "exptime"):
        df[col] = pd.to_numeric(df[col].replace(missing), errors="coerce")

    sources = []
    errors = []

    for idx, row in df.iterrows():
        name = str(row["name"]).strip()
        try:
            if pd.isna(row["ra"]) or pd.isna(row["dec"]):
                raise ValueError("RA and Dec must be decimal degrees.")

            exptime = default_exptime if pd.isna(row["exptime"]) else float(row["exptime"])
            sources.append(SkySource(
                name=name,
                ra_deg=float(row["ra"]),
                dec_deg=float(row["dec"]),
                magnitude=None if pd.isna(row["magnitude"]) else float(row["magnitude"]),
                exptime=exptime,
            ))

        except Exception as exc:
            errors.append(f"Row {idx + 1} ({name!r}): {exc} — skipped.")

    if errors:
        print(f"[WARN] {len(errors)} row(s) skipped while loading {filepath!r}:")
        for error in errors:
            print(f"       {error}")

    print(f"[INFO] Loaded {len(sources)} source(s) from {filepath!r}.")
    return sources

# -----------------------------------------------------------------------------
# Observatory loader
# -----------------------------------------------------------------------------

def load_observatory(name: str, json_path: str = "observatories.json") -> Observatory:
    """
    Load an observatory by name key from a JSON file.

    Expected JSON structure
    -----------------------
    {
      "Siding Spring": {
        "latitude":  -31.2749,
        "longitude": 149.0669,
        "elevation": 1165,
        "timezone":  "Australia/Sydney"
      },
      "Mauna Kea": {
        "latitude":  19.8208,
        "longitude": -155.4681,
        "elevation": 4205,
        "timezone":  "Pacific/Honolulu"
      }
    }

    The "timezone" value must be a valid IANA timezone name.
    See https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Observatory file not found: {json_path!r}")
    except json.JSONDecodeError as exc:
        sys.exit(f"[ERROR] Cannot parse {json_path!r}: {exc}")

    if name not in data:
        available = ", ".join(sorted(data.keys()))
        sys.exit(
            f"[ERROR] Observatory {name!r} not found in {json_path!r}.\n"
            f"        Available: {available}"
        )

    entry = data[name]
    try:
        return Observatory(
            name      = name,
            latitude  = float(entry["latitude"]),
            longitude = float(entry["longitude"]),
            elevation = float(entry.get("elevation", 0.0)),
            timezone  = entry.get("timezone", "UTC"),
        )
    except KeyError as exc:
        sys.exit(f"[ERROR] Observatory {name!r} is missing field {exc} in {json_path!r}.")
    except ValueError as exc:
        sys.exit(f"[ERROR] Observatory {name!r}: {exc}")

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="zeniths.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"{APP_NAME} — {APP_EXPANSION}\n"
            "-------------------------------------------------------------\n"
            f"{APP_TAGLINE}. Computes target visibility, constructs optimized multi-night\n"
            "observing sequences, fills unused gaps, and globally balances observing time.\n"
        ),
        epilog=(
            "Example usage:\n"
            "  python zeniths.py \\\n"
            "    --sources targets.txt \\\n"
            "    --observatory 'Siding Spring' \\\n"
            "    --starttime 2025-06-15T18:00 \\\n"
            "    --endtime 2025-06-30T06:00 \\\n"
            "    --min-alt 20 \\\n"
            "    --moon-excl 25 \\\n"
            "    --max-airmass 2.5 \\\n"
            "    --step-minutes 1.0 \\\n"
            "    --exptime 40 \\\n"
            "    --min-window 20 \\\n"
            "    --overhead 5 \\\n"
            "    --nights 5 \\\n"
            "    --output zeniths_schedule \\\n"
            "    --second-axis ut1 \\\n"
            "    -plot \\\n"
        ),
    )

    parser.add_argument(
        "--sources", "-s",
        metavar="FILE",
        required=True,
        help="Path to the plain-text file listing RA/Dec targets.",
    )

    parser.add_argument(
        "--observatory", "-o",
        metavar="NAME",
        required=True,
        help="Observatory name key in observatories.json.",
    )

    parser.add_argument(
        "--starttime", "-st",
        metavar="LOCAL_ISO",
        required=True,
        help="Start of the observation window in LOCAL time (e.g. 2025-06-15T18:00).",
    )

    parser.add_argument(
        "--endtime", "-et",
        metavar="LOCAL_ISO",
        required=True,
        help="End of the observation window in LOCAL time (e.g. 2025-06-16T06:00).",
    )

    parser.add_argument(
        "--nights", "-n",
        metavar="N",
        type=int,
        required=True,
        default=None,
        help=(
            "Number of nights to use for the observation plan. "
        ),
    )

    # ======================================================================================
    # Optional constraints for visibility calculation and scheduling
    parser.add_argument(
        "--min-alt",
        metavar="DEG",
        type=float,
        default=45.0,
        help="Minimum altitude (degrees) for a usable observation (default: 45).",
    )
    parser.add_argument(
        "--moon-excl",
        metavar="DEG",
        type=float,
        default=60.0,
        help="Minimum Moon-source angular separation in degrees (default: 60).",
    )

    parser.add_argument(
        "--twilight", type=float, default=-18.0,
        help="Sun altitude defining the usable observing night in degrees (default: -18).",
    )

    parser.add_argument(
        "--max-airmass",
        metavar="X",
        type=float,
        default=3.0,
        help="Airmass above which observations are flagged (default: 3.0).",
    )
    parser.add_argument(
        "--step-minutes",
        metavar="INT",
        type=float,
        default=5.0,
        help="Sampling interval in minutes for source tracking (default: 5.0)."
    )
    parser.add_argument(
        "--exptime",
        metavar="MIN",
        type=float,
        default=40.0,
        help="Default required exposure time per source when the table value is missing (default: 40 min).",
    )
    parser.add_argument(
        "--min-window",
        metavar="MIN",
        type=float,
        default=20.0,
        help="Minimum duration of any continuous observing window (default: 20 min).",
    )
    parser.add_argument(
        "--overhead",
        metavar="MIN",
        type=float,
        default=10.0,
        help="Overhead minutes between consecutive target switches (default: 10).",
    )
    parser.add_argument(
        "--bright-time",
        action="store_true",
        help="Allow unrestricted bright-time scheduling: ignore Moon separation and illumination when optimizing.",
    )

    parser.add_argument(
        "--output",
        metavar="FILE",
        required=True,
        help="Output basename (e.g. zeniths_schedule -> zeniths_schedule.tsv and, with -plot, .pdf).",
    )

    parser.add_argument(
        "-plot",
        default=False,
        action="store_true",
        help="Save ZENITHS night charts to <output>.pdf. If omitted, no PDF is generated.",
    )

    parser.add_argument(
        "--second-axis", 
        choices=["lst", "ut1", "utc"],
        default="lst",
        help="Secondary axis for plotting (default: lst)."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    print("\n" + "═" * 78)
    print(f"{APP_NAME} — {APP_EXPANSION}")
    print(APP_TAGLINE)
    print("═" * 78 + "\n")

    # -- Load observatory -----------------------------------------------------
    obs = load_observatory(args.observatory)

    print(f"  Observatory : {obs.name}")
    print(f"  Location    : lat {obs.latitude:+.4f}°   lon {obs.longitude:+.4f}°   "
          f"elev {obs.elevation:.0f} m")
    print(f"  Timezone    : {obs.timezone}\n")

    # -- Parse local start/end times → UTC ------------------------------------
    try:
        start_local = parse_local_datetime(args.starttime)
        end_local   = parse_local_datetime(args.endtime)
    except ValueError as exc:
        parser.error(str(exc))

    start_utc = obs.local_to_utc(start_local)
    end_utc   = obs.local_to_utc(end_local)

    if end_utc <= start_utc:
        parser.error("--endtime must be later than --starttime.")

    if args.nights is not None and args.nights <= 0:
        parser.error("--nights must be a positive integer.")
    if args.exptime <= 0 or args.min_window <= 0 or args.step_minutes <= 0:
        parser.error("--exptime, --min-window and --step-minutes must be positive.")
    if args.overhead < 0:
        parser.error("--overhead cannot be negative.")

    print(f"  Window      : {start_local.strftime('%Y-%m-%d %H:%M')} → "
          f"{end_local.strftime('%Y-%m-%d %H:%M')} local")
    print(f"            = : {start_utc.strftime('%Y-%m-%d %H:%M')} → "
          f"{end_utc.strftime('%Y-%m-%d %H:%M')} UTC\n")

    # -- Load sources ---------------------------------------------------------
    try:
        targets = load_sources_from_file(args.sources, args.exptime)
    except FileNotFoundError:
        parser.error(f"[ERROR] Source file not found: {args.sources!r}")

    if not targets:
        parser.error("[ERROR] No valid sources were loaded from the file. Aborting.")
    
    unique_names = set()
    for src in targets:
        if src.name in unique_names:
            parser.error(f"[ERROR] Duplicate source name found: {src.name!r}. "
                         "Source names must be unique.")
        unique_names.add(src.name)

    if end_local - start_local > timedelta(days=90):
            parser.error("[ERROR] The time window is too long (>90 days). "
                            "Please narrow it down to a more reasonable range.")
    elif end_local - start_local > timedelta(days=30):
        print("[WARN] The time window is very long (>30 days). This may result in long computation times. " 
                "Consider narrowing the window if you only need a specific night or two.")
    elif end_local - start_local < timedelta(hours=6):
        parser.error("[ERROR] The time window must be at least 6 hours long to cover a full night "
                        "and allow for meaningful visibility calculations.")
    
    # -- Build ZENITHS scheduler ----------------------------------------------
    calc = VisibilityCalculator(
        observatory        = obs,
        sources            = targets,
        min_altitude       = args.min_alt,
        moon_exclusion_deg = args.moon_excl,
        max_airmass        = args.max_airmass,
        bright_time        = args.bright_time,
    )

    step_minutes = args.step_minutes
    print(f"[INFO] Sampling window with {step_minutes:.1f}-minutes cadence")
    calc.generate_tracking_data(
        start_utc=start_utc,
        end_utc=end_utc,
        step_minutes=step_minutes,
        sun_alt_limit=args.twilight,
    )

    # -- Rank nights and build observation sequences --------------------------
    print(
        f"[INFO] Building observation sequences (default_exptime={args.exptime:.0f} min, "
        f"min_window={args.min_window:.0f} min, overhead={args.overhead:.0f} min) …"
    )

    calc.build_observation_plan(
        exptime_minutes=args.exptime,
        min_window_minutes=args.min_window,
        overhead_minutes=args.overhead,
        step_minutes=step_minutes,
        max_nights=args.nights,
        sun_alt_limit=args.twilight,
    )

    calc.write_observation_report(
        start_utc          = start_utc,
        end_utc            = end_utc,
        min_altitude       = args.min_alt,
        moon_exclusion_deg = args.moon_excl,
        max_airmass        = args.max_airmass,
        exptime_minutes    = args.exptime,
        min_window_minutes = args.min_window,
        overhead_minutes   = args.overhead,
        step_minutes       = step_minutes,
        sun_alt_limit      = args.twilight,
        output_path        = args.output,
    )

    # -- Altitude-track plots — optimized global plan --------------------------
    if args.plot:
        calc.plot_tracking(
            min_altitude    = args.min_alt,
            save_path       = args.output,
            second_axis     = args.second_axis,
        )

    else:
        print("[INFO] No -plot specified; skipping plot.")


if __name__ == "__main__":
    main()
