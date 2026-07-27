"""Analysis helpers for paired, full, and fixed-SCM simulation designs."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


DEFAULT_REPLICATE_KEYS = (
    "experiment_id",
    "scm_replicate_id",
    "data_replicate_id",
)


def condition_key(row: pd.Series | dict, contrasted: Iterable[str] = ()) -> tuple:
    """Return the non-contrasted condition identity for a result row."""
    excluded = set(contrasted)
    return tuple((k, row[k]) for k in sorted(row) if k not in excluded)


def pair_deltas(
    results: pd.DataFrame,
    reference: dict,
    comparison: dict,
    *,
    value_column: str = "metric_value",
    keys: Iterable[str] = (),
) -> pd.DataFrame:
    """Join reference and comparison rows and compute signed deltas."""
    keys = list(keys) or [*DEFAULT_REPLICATE_KEYS, "target_effect_id"]
    ref = results.loc[_condition_mask(results, reference)].copy()
    comp = results.loc[_condition_mask(results, comparison)].copy()
    ref = ref[keys + [value_column]].rename(columns={value_column: "reference"})
    comp = comp[keys + [value_column]].rename(columns={value_column: "comparison"})
    joined = ref.merge(comp, on=keys, how="inner", validate="one_to_one")
    joined["delta"] = joined["comparison"] - joined["reference"]
    return joined


def _condition_mask(frame: pd.DataFrame, condition: dict) -> pd.Series:
    """Build a boolean mask for a condition dictionary."""
    mask = pd.Series(True, index=frame.index)
    for key, value in condition.items():
        if key not in frame:
            raise KeyError(key)
        mask &= frame[key].eq(value)
    return mask


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")


def _unit_values(
    results: pd.DataFrame,
    *,
    value_column: str,
    replicate_keys: list[str],
) -> pd.DataFrame:
    """Collapse result rows to one metric value per replicate unit."""
    _require_columns(results, [*replicate_keys, value_column])
    values = results[[*replicate_keys, value_column]].copy()
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    values = values.dropna(subset=[value_column])
    if values.empty:
        raise ValueError("No numeric metric values are available for bootstrapping")
    return values.groupby(replicate_keys, as_index=False, dropna=False)[value_column].mean()


def bootstrap_replicates(
    results: pd.DataFrame,
    *,
    group: str | None = None,
    design_mode: str = "full",
    replicate_keys: Iterable[str] = DEFAULT_REPLICATE_KEYS,
    iterations: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Resample completed simulation units for a design-aware bootstrap.

    ``full`` samples complete SCM/data replicate units. ``fixed_scm`` samples
    data replicates within every SCM and never resamples SCMs, which preserves
    the conditional-on-SCM interpretation. ``group`` is retained as a
    compatibility shortcut for flat, single-level data.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if design_mode not in {"full", "fixed_scm"}:
        raise ValueError("design_mode must be 'full' or 'fixed_scm'")
    if group is not None:
        replicate_keys = [group]
    else:
        replicate_keys = list(replicate_keys)
    _require_columns(results, replicate_keys)
    rng = np.random.default_rng(seed)
    units = results.dropna(subset=replicate_keys).copy()
    if units.empty:
        return results.iloc[0:0].copy()

    draws: list[pd.DataFrame] = []
    if design_mode == "full" or len(replicate_keys) == 1:
        keys = units[replicate_keys].drop_duplicates().to_records(index=False)
        selected = np.arange(len(keys))
        for iteration in range(iterations):
            chosen = rng.choice(selected, size=len(selected), replace=True)
            for position, index in enumerate(chosen):
                key = keys[index]
                mask = np.ones(len(units), dtype=bool)
                for column, value in zip(replicate_keys, key, strict=True):
                    mask &= units[column].eq(value).to_numpy()
                rows = units.loc[mask].copy()
                rows["bootstrap_iteration"] = iteration
                rows["bootstrap_draw_position"] = position
                rows["bootstrap_scm_draw_position"] = position
                rows["bootstrap_data_draw_position"] = 0
                draws.append(rows)
    else:
        scm_key = "scm_replicate_id"
        data_key = "data_replicate_id"
        _require_columns(units, [scm_key, data_key])
        for iteration in range(iterations):
            for scm_position, (_, scm_rows) in enumerate(units.groupby(scm_key, dropna=False)):
                data_ids = scm_rows[data_key].drop_duplicates().to_numpy()
                chosen = rng.choice(data_ids, size=len(data_ids), replace=True)
                for data_position, data_id in enumerate(chosen):
                    rows = scm_rows.loc[scm_rows[data_key].eq(data_id)].copy()
                    rows["bootstrap_iteration"] = iteration
                    rows["bootstrap_draw_position"] = data_position
                    rows["bootstrap_scm_draw_position"] = scm_position
                    rows["bootstrap_data_draw_position"] = data_position
                    draws.append(rows)
    return pd.concat(draws, ignore_index=True) if draws else results.iloc[0:0].copy()


def bootstrap_confidence_interval(
    results: pd.DataFrame,
    *,
    value_column: str = "metric_value",
    design_mode: str = "full",
    replicate_keys: Iterable[str] = DEFAULT_REPLICATE_KEYS,
    iterations: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, object]:
    """Compute a percentile CI from completed simulation replicate metrics.

    Full mode resamples complete SCM/data units, so the interval includes SCM
    and observational-generation variation. Fixed-SCM mode resamples data
    units independently within each SCM and averages the SCM-level statistics,
    so it measures observational-generation variation conditional on the
    observed SCM blocks.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    keys = list(replicate_keys)
    values = _unit_values(results, value_column=value_column, replicate_keys=keys)
    draws = bootstrap_replicates(
        values,
        design_mode=design_mode,
        replicate_keys=keys,
        iterations=iterations,
        seed=seed,
    )
    bootstrap_values = []
    if design_mode == "fixed_scm" and "scm_replicate_id" in keys:
        for _, iteration_rows in draws.groupby("bootstrap_iteration", sort=True):
            scm_statistics = [
                statistic(group[value_column].to_numpy(dtype=float))
                for _, group in iteration_rows.groupby("scm_replicate_id", dropna=False)
            ]
            bootstrap_values.append(float(np.mean(scm_statistics)))
    else:
        for _, iteration_rows in draws.groupby("bootstrap_iteration", sort=True):
            bootstrap_values.append(
                float(statistic(iteration_rows[value_column].to_numpy(dtype=float)))
            )
    alpha = (1 - confidence) / 2
    interval = percentile_interval(bootstrap_values, confidence=confidence)
    return {
        "estimate": float(statistic(values[value_column].to_numpy(dtype=float))),
        "lower": interval[0],
        "upper": interval[1],
        "confidence": confidence,
        "method": "percentile_bootstrap",
        "design_mode": design_mode,
        "iterations": iterations,
        "bootstrap_values": np.asarray(bootstrap_values),
    }


def percentile_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return a percentile interval, preserving NaN behavior for empty input."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return (float("nan"), float("nan"))
    tail = (1 - confidence) / 2
    return tuple(np.quantile(values, [tail, 1 - tail]))
