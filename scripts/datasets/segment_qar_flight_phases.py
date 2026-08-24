# -*- coding: utf-8 -*-
"""Segment QAR flights into explainable flight phases.

The implementation follows the project decision hierarchy: decode altitude,
prefer AIR/GROUND when present, use smoothed vertical trend and engine power
as supporting evidence, and keep post-TOD level-offs inside Descent.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASES = ("Taxi-out", "Takeoff", "Initial Climb", "Climb", "Cruise", "Descent", "Approach", "Landing", "Taxi-in")
PHASE_COLORS = {
    "Taxi-out": "#8c8c8c", "Takeoff": "#d62728", "Initial Climb": "#ff9896",
    "Climb": "#2ca02c", "Cruise": "#1f77b4", "Descent": "#9467bd",
    "Approach": "#e377c2", "Landing": "#ff7f0e", "Taxi-in": "#7f7f7f",
}
QAR_RE = re.compile(r"^(?P<aircraft>B-[^_]+)_(?P<timestamp>\d{14})\.qar\.csv$", re.IGNORECASE)


def _numeric(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    upper = {str(column).upper(): column for column in df.columns}
    for name in names:
        key = name.upper()
        if key in upper:
            return pd.to_numeric(df[upper[key]], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _mean_matching(df: pd.DataFrame, patterns: tuple[str, ...]) -> pd.Series:
    selected = []
    for column in df.columns:
        upper = str(column).upper()
        if any(re.search(pattern, upper) for pattern in patterns):
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().sum() >= max(3, int(len(df) * 0.01)):
                selected.append(values)
    if not selected:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.concat(selected, axis=1).median(axis=1, skipna=True)


def _fill_short(values: pd.Series, limit: int = 30) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan).interpolate(limit=limit, limit_direction="both")


def _run_length(mask: np.ndarray, minimum: int) -> tuple[int | None, int | None]:
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value and index == len(mask) - 1 else index - 1
            if end - start + 1 >= minimum:
                return start, end
            start = None
    return None, None


def _last_run(mask: np.ndarray, minimum: int) -> tuple[int | None, int | None]:
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value and index == len(mask) - 1 else index - 1
            if end - start + 1 >= minimum:
                runs.append((start, end))
            start = None
    return runs[-1] if runs else (None, None)


def _height_and_signals(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    b2694 = "ALTITUDE_STD" in df.columns or "AIR/GROUND" in df.columns
    if b2694:
        height = _numeric(df, ["ALTITUDE_STD"])
        air_ground = _numeric(df, ["AIR/GROUND"])
        # QAR variants sometimes use 0/1 and sometimes text-decoded numeric values.
        air_ground = (air_ground >= 0.5).astype(float).where(air_ground.notna())
        ground_speed = _numeric(df, ["GS", "GROUND_SPEED"])
        speed = _numeric(df, ["CAS", "IAS", "MACH"])
        power = _mean_matching(df, (r"SELECTED_N1", r"^N1[12]$", r"N1_CMD", r"TARGET_N1"))
        fuel_flow = _mean_matching(df, (r"SELECTED_FUEL_FLOW", r"^FF[12]$", r"FUEL_FLOW"))
        aircraft_type = "B-2694"
    else:
        raw_alt = _numeric(df, ["ALT"])
        corrected = raw_alt.where(raw_alt >= -1000, raw_alt + 65536.0)
        height = corrected * 2.0
        air_ground = pd.Series(np.nan, index=df.index, dtype=float)
        ground_speed = _numeric(df, ["GS"])
        speed = _numeric(df, ["CAS", "MACH"])
        power = _mean_matching(df, (r"^N1[12]$", r"^N2[12]$"))
        fuel_flow = _mean_matching(df, (r"^FF[12]$", r"FUEL_FLOW"))
        aircraft_type = "B-1400"

    height = _fill_short(height)
    smoothed = height.rolling(31, center=True, min_periods=5).median().bfill().ffill()
    # A centered 60-second slope, expressed in ft/min because QAR rows are 1 Hz.
    vertical_speed = (smoothed.shift(-30) - smoothed.shift(30))
    vertical_speed = vertical_speed.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    out = pd.DataFrame({
        "height_ft": height,
        "vertical_speed_fpm": vertical_speed,
        "air_ground": air_ground,
        "ground_speed_kt": ground_speed,
        "speed_indicator": speed,
        "engine_power": power,
        "fuel_flow": fuel_flow,
    })
    return aircraft_type, out


def _airport_levels(signals: pd.DataFrame, aircraft_type: str) -> tuple[float, float]:
    n = len(signals)
    first = signals.iloc[: min(300, n)]
    last = signals.iloc[max(0, n - min(300, n)):]
    if aircraft_type == "B-2694" and signals["air_ground"].notna().any():
        dep_values = first.loc[first["air_ground"] < 0.5, "height_ft"].dropna()
        arr_values = last.loc[last["air_ground"] < 0.5, "height_ft"].dropna()
    else:
        dep_values = first.loc[first["ground_speed_kt"].fillna(999) < 40, "height_ft"].dropna()
        arr_values = last.loc[last["ground_speed_kt"].fillna(999) < 40, "height_ft"].dropna()
    dep = float(dep_values.median()) if len(dep_values) else float(signals["height_ft"].iloc[0])
    arr = float(arr_values.median()) if len(arr_values) else float(signals["height_ft"].iloc[-1])
    return dep, arr


def segment_flight(df: pd.DataFrame, aircraft_type: str) -> tuple[pd.DataFrame, dict]:
    signals = _height_and_signals(df)[1]
    n = len(signals)
    dep_level, arr_level = _airport_levels(signals, aircraft_type)
    signals["height_agl_departure_ft"] = signals["height_ft"] - dep_level
    signals["height_agl_arrival_ft"] = signals["height_ft"] - arr_level

    ag = signals["air_ground"].to_numpy()
    liftoff = None
    touchdown = None
    if aircraft_type == "B-2694" and np.isfinite(ag).any():
        transitions_up = np.flatnonzero((ag[1:] >= 0.5) & (ag[:-1] < 0.5)) + 1
        transitions_down = np.flatnonzero((ag[1:] < 0.5) & (ag[:-1] >= 0.5)) + 1
        liftoff = int(transitions_up[0]) if len(transitions_up) else None
        touchdown = int(transitions_down[-1]) if len(transitions_down) else None
    if liftoff is None:
        candidate = (signals["ground_speed_kt"].fillna(0).to_numpy() > 100) & (signals["vertical_speed_fpm"].fillna(0).to_numpy() > 300)
        start, _ = _run_length(candidate, 5)
        liftoff = start
    if touchdown is None:
        candidate = (signals["height_agl_arrival_ft"].abs().to_numpy() < 150) & (signals["ground_speed_kt"].fillna(999).to_numpy() < 80) & (np.arange(n) > int(n * 0.5))
        touchdown, _ = _run_length(candidate, 5)

    liftoff = int(liftoff) if liftoff is not None else max(1, int(n * 0.08))
    touchdown = int(touchdown) if touchdown is not None and touchdown > liftoff else max(liftoff + 1, int(n * 0.92))

    power = signals["engine_power"].to_numpy()
    high_power = np.nanmedian(power[max(0, liftoff - 60): liftoff]) if np.isfinite(power[max(0, liftoff - 60): liftoff]).any() else np.nan
    fuel = signals["fuel_flow"].to_numpy()
    high_fuel = np.nanmedian(fuel[max(0, liftoff - 60): liftoff]) if np.isfinite(fuel[max(0, liftoff - 60): liftoff]).any() else np.nan
    initial_end = None
    if np.isfinite(high_power) or np.isfinite(high_fuel):
        power_reduction = power < high_power * 0.97 if np.isfinite(high_power) else np.zeros(n, dtype=bool)
        fuel_reduction = fuel < high_fuel * 0.90 if np.isfinite(high_fuel) else np.zeros(n, dtype=bool)
        reduction = power_reduction | fuel_reduction
        candidate = reduction & (np.arange(n) > liftoff + 20) & (np.arange(n) < liftoff + 1200) & (signals["height_agl_departure_ft"].to_numpy() > 1000)
        initial_end, _ = _run_length(candidate, 10)
    if initial_end is None:
        above_1500 = signals["height_agl_departure_ft"].to_numpy() > 1500
        initial_end, _ = _run_length(above_1500 & (np.arange(n) > liftoff), 10)
    initial_end = int(initial_end) if initial_end is not None else min(liftoff + 180, touchdown - 1)

    h = signals["height_ft"].to_numpy()
    vs = signals["vertical_speed_fpm"].to_numpy()
    cruise_level = float(np.nanpercentile(h[liftoff:touchdown], 95)) if touchdown > liftoff else float(np.nanpercentile(h, 95))
    cruise_candidate = (h >= 0.80 * cruise_level) & (np.abs(vs) < 150) & (np.arange(n) > initial_end)
    cruise_start, _ = _run_length(cruise_candidate, 180)
    cruise_start = int(cruise_start) if cruise_start is not None else min(initial_end + 1, touchdown - 1)

    tod = None
    for start in range(max(cruise_start + 180, 1), max(touchdown - 60, 1)):
        if vs[start:start + 30].mean() < -300 and h[min(start + 300, n - 1)] < h[start] - 2000:
            tod = start
            break
    tod = int(tod) if tod is not None else max(cruise_start + 180, int(touchdown * 0.65))

    approach_candidate = (signals["height_agl_arrival_ft"].to_numpy() < 5000) & (np.arange(n) > tod) & (np.arange(n) < touchdown)
    if aircraft_type == "B-1400":
        approach_candidate &= (vs < 50)
    approach_start, _ = _run_length(approach_candidate, 30)
    approach_start = int(approach_start) if approach_start is not None else max(tod + 1, touchdown - 600)

    labels = np.full(n, "Descent", dtype=object)
    labels[: max(0, liftoff - 30)] = "Taxi-out"
    labels[max(0, liftoff - 30): liftoff] = "Takeoff"
    labels[liftoff: initial_end] = "Initial Climb"
    labels[initial_end: cruise_start] = "Climb"
    labels[cruise_start: tod] = "Cruise"
    labels[tod: approach_start] = "Descent"
    labels[approach_start: touchdown] = "Approach"
    labels[touchdown: min(n, touchdown + 60)] = "Landing"
    labels[min(n, touchdown + 60):] = "Taxi-in"

    vertical_state = np.where(vs > 300, "CLM", np.where(vs < -300, "DES", "LEV"))
    power_state = np.full(n, "UNKNOWN", dtype=object)
    if np.isfinite(power).any():
        q = np.nanpercentile(power, [25, 60, 85])
        power_state = np.where(power >= q[2], "HIGH", np.where(power <= q[0], "LOW", "MID"))
    result = signals.copy()
    result["flight_phase"] = labels
    result["vertical_state"] = vertical_state
    result["power_state"] = power_state
    result["phase_confidence"] = np.where(result["flight_phase"].isin(["Cruise", "Descent"]), 0.85, 0.70)
    result["phase_confidence"] = result["phase_confidence"].where(result["air_ground"].isna() | result["flight_phase"].isin(["Takeoff", "Landing"]), 0.95)
    result["segment_id"] = (result["flight_phase"] != result["flight_phase"].shift()).cumsum().astype(int)
    metrics = {
        "n_rows": n, "aircraft_type": aircraft_type, "departure_level_ft": dep_level, "arrival_level_ft": arr_level,
        "liftoff_index": liftoff, "initial_climb_end_index": initial_end, "cruise_start_index": cruise_start,
        "top_of_descent_index": tod, "approach_start_index": approach_start, "touchdown_index": touchdown,
        "cruise_height_reference_ft": cruise_level,
        "cruise_rows": int((labels == "Cruise").sum()), "phase_transition_count": int((labels[1:] != labels[:-1]).sum()),
        "missing_height_ratio": float(signals["height_ft"].isna().mean()),
        "air_ground_available": bool(signals["air_ground"].notna().any()),
        "phase_duration_seconds": {phase: int((labels == phase).sum()) for phase in PHASES},
    }
    return result, metrics


def _plot_flight(result: pd.DataFrame, metrics: dict, output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    x = np.arange(len(result))
    for phase, color in PHASE_COLORS.items():
        mask = result["flight_phase"].to_numpy() == phase
        axes[0].scatter(x[mask], result.loc[mask, "height_ft"], s=3, color=color, label=phase, rasterized=True)
    axes[0].axhline(metrics["cruise_height_reference_ft"], color="#1f77b4", linestyle="--", linewidth=1, label="Cruise reference")
    axes[0].set_ylabel("Decoded pressure height (ft)")
    axes[0].set_title(title)
    axes[0].legend(ncol=5, fontsize=8, loc="upper center")
    axes[0].grid(alpha=0.2)
    axes[1].plot(x, result["vertical_speed_fpm"], color="#333333", linewidth=0.7)
    axes[1].axhline(300, color="#2ca02c", linestyle=":")
    axes[1].axhline(-300, color="#d62728", linestyle=":")
    axes[1].set_ylabel("VS (ft/min)")
    axes[1].set_xlabel("QAR row index (1 second per row)")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_cruise_distribution(all_metrics: list[dict], output: Path) -> None:
    frame = pd.DataFrame(all_metrics)
    if frame.empty:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in frame.groupby("aircraft_type"):
        ax.scatter(np.full(len(group), label), group["cruise_height_reference_ft"], alpha=0.55, s=18, label=label)
    ax.set_ylabel("Cruise reference height (ft)")
    ax.set_xlabel("Aircraft family")
    ax.set_title("Cruise-height reference after flight-phase segmentation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(input_dir: Path, output_dir: Path, plot_files: list[str] | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    selected = set(plot_files or [])
    for path in sorted(input_dir.glob("*.qar.csv")):
        match = QAR_RE.match(path.name)
        if not match:
            continue
        df = pd.read_csv(path, low_memory=False)
        aircraft_type, _ = _height_and_signals(df)
        result, item = segment_flight(df, aircraft_type)
        item.update({"source_file": path.name, "aircraft_id": match.group("aircraft"), "source_timestamp": match.group("timestamp")})
        metrics.append(item)
        result.insert(0, "source_file", path.name)
        result.to_csv(segment_dir / path.name.replace(".qar.csv", ".segments.csv"), index=False)
        if path.name in selected:
            _plot_flight(result, item, output_dir / "plots" / f"{path.stem}_cruise_height.png", f"{path.name}: cruise-height segmentation")
    _plot_cruise_distribution(metrics, output_dir / "plots" / "cruise_height_distribution.png")
    summary = {
        "schema_version": 1,
        "method": "altitude_decode_plus_vertical_trend_plus_power_plus_air_ground_state_machine",
        "sampling_interval": "1 second per QAR row",
        "phase_order": list(PHASES),
        "thresholds": {"vertical_state_climb_fpm": 300, "vertical_state_descent_fpm": -300, "level_abs_fpm": 150, "cruise_min_seconds": 180, "approach_agl_ft": 5000},
        "source_count": len(metrics),
        "source_count_by_aircraft": {str(key): int(value) for key, value in pd.Series([item["aircraft_id"] for item in metrics]).value_counts().items()},
        "metrics": metrics,
    }
    (output_dir / "segmentation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(metrics).to_csv(output_dir / "segmentation_evaluation.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment QAR flights and visualize cruise-height results")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--output-dir", default="data/aero_engine_dataset/flight_phase_segments")
    parser.add_argument("--plot-file", action="append", default=[])
    args = parser.parse_args()
    summary = run(Path(args.input_dir), Path(args.output_dir), args.plot_file)
    print(json.dumps({"source_count": summary["source_count"], "source_count_by_aircraft": summary["source_count_by_aircraft"], "output_dir": args.output_dir}, ensure_ascii=False))


if __name__ == "__main__":
    main()
