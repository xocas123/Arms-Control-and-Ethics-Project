"""
Shared temporal analysis utilities.
Used by all four research question modules.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

MILESTONES = {
    1963: "PTBT signed",
    1968: "NPT opened",
    1972: "BWC + ABM + SALT I",
    1987: "INF Treaty",
    1991: "End of Cold War / START I",
    1993: "CWC signed",
    1996: "ICJ Advisory Opinion / CTBT",
    1997: "Ottawa Convention",
    2002: "US withdraws from ABM",
    2005: "NPT RevCon fails",
    2008: "Convention on Cluster Munitions",
    2010: "New START / NPT RevCon action plan",
    2013: "Oslo conference / ATT adopted",
    2014: "Nayarit + Vienna conferences",
    2017: "TPNW adopted",
    2019: "US withdraws from INF",
    2021: "TPNW enters into force",
    2022: "Russia invades Ukraine / Russia suspends New START",
    2023: "New START suspension deepens",
}


def rolling_mean(series: pd.Series, window: int = 5) -> pd.Series:
    """Rolling mean with min_periods=1."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def rolling_std(series: pd.Series, window: int = 5) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window=window, min_periods=2, center=True).std()


def year_over_year_delta(series: pd.Series) -> pd.Series:
    """Year-over-year first difference."""
    return series.diff()


def compute_change_points(
    series: pd.Series,
    method: str = "pelt",
    penalty: Optional[float] = None,
    n_bkps: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """
    Detect structural breaks in time series.

    Returns list of (year, magnitude) tuples where year is the break year
    and magnitude is the change in mean before vs after the break.
    Falls back to variance-based detection if ruptures is not available.
    """
    if len(series) < 10:
        return []

    values = series.dropna().values
    index = series.dropna().index

    try:
        import ruptures as rpt

        signal = values.reshape(-1, 1)
        if method == "pelt":
            algo = rpt.Pelt(model="rbf").fit(signal)
            if penalty is not None:
                pen = penalty
            else:
                # Conservative BIC-style penalty: scales with both n and signal variance.
                # Prevents over-segmentation on smooth series (e.g. finding breaks every 5 years).
                n = len(values)
                pen = np.log(n) * np.var(values) * n / 10
            breakpoints = algo.predict(pen=pen)
        else:  # binseg
            n = n_bkps if n_bkps is not None else 4
            algo = rpt.Binseg(model="rbf").fit(signal)
            breakpoints = algo.predict(n_bkps=n)

        # Convert sample indices to years
        results = []
        prev = 0
        for bp in breakpoints[:-1]:  # last element is len(signal)
            if bp >= len(index):
                continue
            year = index[bp] if hasattr(index, "__getitem__") else int(index[bp])
            before_mean = values[prev:bp].mean() if bp > prev else np.nan
            after_mean = values[bp:].mean() if bp < len(values) else np.nan
            magnitude = float(after_mean - before_mean) if not np.isnan(before_mean) else 0.0
            results.append((int(year), magnitude))
            prev = bp
        return results

    except ImportError:
        # Fallback: simple variance-based detection
        return _simple_change_points(series)


def _simple_change_points(series: pd.Series) -> List[Tuple[int, float]]:
    """Simple change-point detection using rolling variance."""
    values = series.dropna()
    if len(values) < 6:
        return []

    window = max(3, len(values) // 8)
    roll_std = values.rolling(window=window, center=True).std()
    local_maxima = []

    for i in range(1, len(roll_std) - 1):
        if (roll_std.iloc[i] > roll_std.iloc[i - 1] and
                roll_std.iloc[i] > roll_std.iloc[i + 1]):
            year = values.index[i]
            before = values.iloc[max(0, i - window):i].mean()
            after = values.iloc[i:min(len(values), i + window)].mean()
            magnitude = float(after - before)
            local_maxima.append((int(year), magnitude))

    # Keep top 3 by absolute magnitude
    local_maxima.sort(key=lambda x: abs(x[1]), reverse=True)
    return local_maxima[:3]


def normalize_to_event(
    df: pd.DataFrame,
    country_col: str,
    year_col: str,
    event_year_col: str,
    window: int = 10,
) -> pd.DataFrame:
    """
    Reindex time axis so year 0 = event year.
    Filters to ±window years around the event.
    Returns DataFrame with added column 'year_relative'.
    """
    result = df.copy()
    result["year_relative"] = result[year_col] - result[event_year_col]
    return result[result["year_relative"].abs() <= window].copy()


def compute_group_time_series(
    df: pd.DataFrame,
    value_col: str,
    year_col: str = "year",
    group_col: Optional[str] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate a value column by year (and optionally by group).
    Returns DataFrame: year, [group], value_mean, value_std, n.
    """
    group_keys = [year_col]
    if group_col:
        group_keys.append(group_col)

    grouped = df.groupby(group_keys)[value_col]

    result = pd.DataFrame({
        f"{value_col}_mean": grouped.mean(),
        f"{value_col}_std": grouped.std(),
        "n": grouped.count(),
    }).reset_index()

    return result


def compute_between_group_distance(
    group_a_embeddings: np.ndarray,
    group_b_embeddings: np.ndarray,
) -> float:
    """Cosine distance between centroids of two groups of embeddings."""
    if len(group_a_embeddings) == 0 or len(group_b_embeddings) == 0:
        return np.nan
    centroid_a = group_a_embeddings.mean(axis=0)
    centroid_b = group_b_embeddings.mean(axis=0)
    sim = _cosine_sim_1d(centroid_a, centroid_b)
    return float(1.0 - sim)


def compute_within_group_variance(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance within a group (proxy for internal variance)."""
    if len(embeddings) < 2:
        return 0.0
    centroid = embeddings.mean(axis=0)
    sims = np.array([_cosine_sim_1d(e, centroid) for e in embeddings])
    return float(1.0 - sims.mean())


def _cosine_sim_1d(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_linear_slope(series: pd.Series, year_range: Optional[Tuple] = None) -> float:
    """Compute OLS slope of series (optionally restricted to year range)."""
    s = series.dropna()
    if year_range:
        s = s[(s.index >= year_range[0]) & (s.index <= year_range[1])]
    if len(s) < 2:
        return np.nan
    x = np.array(s.index, dtype=float)
    y = s.values.astype(float)
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


def annotate_milestones(
    ax,
    year_range: Optional[Tuple] = None,
    alpha: float = 0.3,
    fontsize: int = 7,
    color: str = "gray",
    selected_milestones: Optional[List[int]] = None,
):
    """
    Add milestone vertical lines to a matplotlib axes object.
    Filters to milestones within year_range.
    """
    milestones = selected_milestones or list(MILESTONES.keys())
    for year, label in MILESTONES.items():
        if year not in milestones:
            continue
        if year_range and not (year_range[0] <= year <= year_range[1]):
            continue
        ax.axvline(x=year, color=color, alpha=alpha, linestyle="--", linewidth=0.8)
        ax.text(
            year, ax.get_ylim()[1] * 0.98, label,
            rotation=90, fontsize=fontsize, color=color, alpha=0.8,
            va="top", ha="right",
        )
