# ZENITHS

**Zenith Exposure and Nighttime Intelligent Target Handling & Scheduling**

ZENITHS is a visibility-aware astronomical observation scheduler for building multi-night target sequences from a source list, observatory geometry, observing constraints, per-target exposure requirements, and telescope overheads.

It is designed for the practical problem of turning a target catalogue into an observing plan: complete as many targets as possible, place their required exposures at favorable times, use otherwise idle time productively, and redistribute the remaining night so difficult targets can receive more clock time than easy targets without changing the selected observing topology.

## Features

- Tracks source altitude/azimuth with Astropy from a named observatory.
- Accepts local observing-window times and performs timezone-aware UTC conversion.
- Applies hard constraints from minimum altitude, maximum airmass, twilight, and Moon separation.
- Uses a smooth positive observing-efficiency score based on target brightness, airmass, twilight depth, and Moon background.
- Supports a per-source exposure time with a command-line fallback value.
- Rounds required exposure upward to the scheduling cadence rather than rejecting non-integer cadence multiples.
- Enforces a minimum duration for every continuous observing window.
- Accounts for target-switch overheads.
- Optimizes across a user-selected maximum number of nights.
- Allows additional observing windows for already-completed sources when gaps remain.
- Globally rebalances Stage-2 windows by jointly moving all separator/overhead anchors.
- Writes a detailed TSV observation report.
- Optionally produces multi-page PDF night charts with the optimized observing windows overlaid on altitude tracks.

## Installation

ZENITHS is currently a single Python script.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install the dependencies directly:

```bash
pip install astropy numpy pandas matplotlib pulp highspy tzdata
```

Python 3.11 or newer is recommended.

HiGHS is used through PuLP for the mixed-integer optimization stages. ZENITHS checks that HiGHS is available before attempting to schedule.

## Repository files

```text
zeniths.py           Main scheduler and CLI
observatories.json   Observatory definitions used by --observatory
requirements.txt     Python dependencies
README.md            This documentation
```

## Source catalogue

The source table can be comma-separated or whitespace-separated. A header is optional.

Recommended format:

```csv
name,ra,dec,magnitude,exptime
J084421.68+215215.4,131.0903,21.8709,20.1,35
J093839.76+272954.6,144.6657,27.4985,19.4,40
J101744.24-033105.2,154.4343,-3.5181,21.0,
```

Columns:

| Column | Meaning |
| --- | --- |
| `name` | Unique source identifier |
| `ra` | Right ascension in **decimal degrees** |
| `dec` | Declination in **decimal degrees** |
| `magnitude` | Optional broadband magnitude used by the observing-efficiency score |
| `exptime` | Required total exposure time in minutes; if missing, `--exptime` is used |

`ra_deg` / `dec_deg` are accepted aliases for `ra` / `dec`, and `mag` is accepted as an alias for `magnitude`.

RA/Dec sexagesimal strings are intentionally not accepted by the current loader.

### Exposure quantization

The scheduler works on the cadence specified by `--step-minutes`. A requested exposure is rounded **up** to the next complete cadence slot:

```text
required_slots = ceil(exptime / step_minutes)
```

For example, `exptime=35` min with `--step-minutes 2` becomes 18 slots = 36 minutes.

## Observatory configuration

`--observatory` selects a key in `observatories.json`.

```json
{
  "keck-1": {
    "latitude": 19.8263,
    "longitude": -155.4744,
    "elevation": 4145,
    "timezone": "Pacific/Honolulu"
  }
}
```

Longitudes are positive east and negative west. `timezone` must be an IANA timezone name so daylight-saving and local/UTC conversion are handled correctly.

## Quick start

For the two-night Keck/MOSFIRE-style scheduling case used during development:

```bash
python zeniths.py \
  --sources mosfire_2027A.csv \
  --observatory keck-1 \
  --starttime 2027-03-15T18:00 \
  --endtime 2027-03-25T06:00 \
  --nights 2 \
  --output mosfire_2027A_vis3 \
  --exptime 35 \
  --max-airmass 1.3 \
  --overhead 10 \
  --min-window 20 \
  --step-minutes 2 \
  --bright-time \
  -plot
```

Outputs:

```text
mosfire_2027A_vis3.tsv
mosfire_2027A_vis3.pdf
```

The PDF is only produced when `-plot` is supplied.

## Command-line options

| Option | Description | Default |
| --- | --- | ---: |
| `--sources`, `-s` | Source table | required |
| `--observatory`, `-o` | Key in `observatories.json` | required |
| `--starttime`, `-st` | Start of requested window in observatory local time | required |
| `--endtime`, `-et` | End of requested window in observatory local time | required |
| `--nights`, `-n` | Maximum number of nights the optimized plan may use | required |
| `--min-alt` | Hard minimum source altitude | 45 deg |
| `--moon-excl` | Hard Moon-source exclusion angle | 60 deg |
| `--twilight` | Hard Sun-altitude boundary defining the usable night | -18 deg |
| `--max-airmass` | Hard maximum airmass | 3.0 |
| `--step-minutes` | Scheduling/tracking cadence | 5 min |
| `--exptime` | Fallback required exposure when source-table value is missing | 40 min |
| `--min-window` | Minimum duration of one continuous source window | 20 min |
| `--overhead` | Minimum target-switch overhead | 10 min |
| `--bright-time` | Ignore Moon exclusion and Moon score penalty | off |
| `--output` | Output basename | required |
| `-plot` | Save optimized night charts to `<output>.pdf` | off |
| `--second-axis` | Secondary plot axis: `lst`, `ut1`, or `utc` | `lst` |
| `--version` | Print ZENITHS version | |

## Visibility and observing-efficiency score

A slot is first tested against hard constraints. If any of the following fail, the slot is unavailable:

- source below the configured minimum altitude;
- airmass above `--max-airmass`;
- Sun above `--twilight`;
- Moon too close while above the horizon, unless `--bright-time` is used.

For otherwise legal slots, ZENITHS uses the positive scheduling score

```text
score = 100 × brightness_eff × airmass_eff × sun_eff × moon_eff
```

The score is an **observing-efficiency proxy**, not an instrument exposure-time calculator.

### Brightness term

The reference magnitude is the median finite magnitude of the input sample:

```text
brightness_eff = 10^[-0.4 (m - m_ref)]
```

and is clipped to `[0.10, 10]` so one target cannot dominate the scheduler purely through the magnitude proxy.

A fainter target therefore accumulates integrated observing score more slowly and can receive more clock time during the balancing stage.

### Airmass term

ZENITHS combines a modest atmospheric-extinction term with a soft `1/X` degradation:

```text
transmission_eff = 10^[-0.4 k (X - 1)]
airmass_eff      = transmission_eff / X
```

with `k = 0.10 mag / airmass` in the current implementation.

### Twilight term

The user-selected twilight altitude remains a hard boundary. Inside that boundary, ZENITHS applies a smooth sky-brightness penalty so a slot immediately after evening twilight or immediately before morning twilight is not treated as equivalent to a fully dark slot.

At the hard twilight boundary:

```text
sun_eff = 0.65
```

It rises with a cosine transition to `1.0` after the Sun is another 6 degrees below the configured twilight limit. With `--twilight -18`, full-dark weighting is reached at approximately `-24 deg` Sun altitude.

### Moon term

When Moon constraints are active, a multiplicative penalty depends on Moon illumination, altitude, and proximity to the target. The hard Moon-exclusion angle remains enforced separately.

With `--bright-time`, both the hard Moon-separation restriction and the Moon score penalty are disabled.

## Scheduling algorithm

ZENITHS uses three stages.

### Stage 1 — complete as many targets as possible

For every target on every usable night, ZENITHS enumerates every visibility-valid contiguous block with duration equal to that target's cadence-rounded required exposure.

**Stage 1A** maximizes the number of completed targets, subject to:

- no overlapping observations;
- target-switch overheads;
- no more than `--nights` selected nights;
- one required-exposure block per source in Stage 1;
- selected nights spanning the sampled evening-to-morning twilight interval.

The maximum completed-target count is then frozen.

**Stage 1B** maximizes the total integrated observing-efficiency score of those required-exposure blocks.

### Stage 2 — fill gaps and absorb residual idle time

Stage 2 preserves the completed source set from Stage 1 and works inside each selected night's remaining gaps.

It repeatedly searches all current gaps for the highest-total-score additional observing window that:

- belongs to a source already completed in Stage 1;
- is fully visibility-valid;
- is at least `--min-window` long;
- respects target-switch overhead on both sides.

This permits multiple observing windows for the same source.

When no further legal `--min-window` filler fits, Stage 2 absorbs smaller residual gaps by extending the adjacent legal windows one cadence slot at a time, preferring the higher-score adjacent slot while retaining the required switching overhead.

The terminal log reports each filler window, its duration, integrated score, mean score rate, residual extension time, and final on-target utilization for each selected night.

### Stage 3 — globally rebalance all separator anchors

Stage 3 does **not** rebuild the observing sequence from scratch.

It freezes the Stage-2 topology:

- selected nights;
- source order;
- number of observing windows;
- the width of every inter-window separator/overhead interval;
- evening and morning twilight endpoints.

Every internal separator anchor is then optimized **jointly**. Moving one separator changes the adjacent windows and can constrain all later separators, so they are solved globally rather than moved greedily one at a time.

Each final window must:

- remain within the same contiguous visibility-valid component as its Stage-2 window;
- remain at least `--min-window` long;
- preserve the fixed separator widths;
- keep each source's total exposure across all of its windows at or above its required `exptime`.

Because twilight endpoints and separator widths are fixed, total on-target time is invariant during Stage 3.

**Stage 3A** minimizes the source-to-source spread in total integrated observing-efficiency score:

```text
max(source integrated score) - min(source integrated score)
```

This intentionally tends to allocate more clock time to targets whose instantaneous observing-efficiency score is lower.

**Stage 3B** freezes the best achievable spread and maximizes total integrated observing-efficiency score among equally balanced solutions.

## Output report

`<output>.tsv` begins with a commented ZENITHS metadata section recording:

- ZENITHS version;
- generation time;
- observatory and timezone;
- requested observing window;
- hard scheduling constraints;
- scoring configuration;
- selected-night summary;
- scheduled and unscheduled source counts.

The tabular section contains, among other fields:

```text
source_name
ra_deg
dec_deg
magnitude
exptime
night_date
dark_start_local
dark_end_local
obs_start_local
obs_end_local
duration_min
is_moon_up
max_moon_illum_frac
min_moon_sep_deg
max_airmass
min_altitude_deg
score
mean_score_rate
```

The PDF night-chart pages and PDF metadata are also labeled with the ZENITHS name.

## Important interpretation notes

- The score is a scheduling heuristic. It is **not** a substitute for an instrument ETC or a physical S/N model.
- The magnitude term is a broadband proxy for target difficulty; a line-flux or instrument-specific model would be preferable when such information is available.
- All scheduling durations are quantized by `--step-minutes`.
- "Twilight-to-twilight" currently means the first and last cadence slots satisfying the twilight criterion. It is exact on the scheduler's sampled time grid, not a continuous root-solved Sun-altitude crossing.
- `--nights N` means the plan may use **at most** `N` nights.
- In `--bright-time` mode the Moon is deliberately ignored by the scheduler, while the Sun/twilight terms remain active.

## Development status

ZENITHS is currently an actively developed research scheduler. The core scheduling and reporting pipeline is contained in `zeniths.py`; interfaces and scoring details may evolve as the instrument model becomes more physical.

## Name

**ZENITHS** stands for **Zenith Exposure and Nighttime Intelligent Target Handling & Scheduling**.
