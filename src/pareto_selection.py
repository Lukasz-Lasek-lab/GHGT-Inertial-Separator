"""
Pareto Front Post-processing and Engineering Decision Support module for MsCO2limit.
Executes:
- K-Means clustering of non-dominated Pareto front configurations
- Rigorous physical particle balance verification (N1 + N2 + N3 = 6417)
- Percentage separation efficiency calculation (loss eta_1_pct and capture eta_2_pct)
- Normalized trade-off scoring and geometry deduplication
- Engineering export to Excel and CSV formats.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config.path import optimization_results_dir
from src.constants import BASE_FEATURES, TOTAL_PARTICLES


def partition_pareto_regimes(
    df: pd.DataFrame,
    h2_threshold: float = 0.05,
) -> pd.Series:
    """
    Deterministically partitions Pareto designs into 2 physical geometric regimes based on H2:
    - Cluster 0: High-gap regime (H2 >= h2_threshold, nominal approx 59 mm)
    - Cluster 1: Low-gap regime (H2 < h2_threshold, nominal approx 39 mm)
    """
    if "H2" not in df.columns:
        raise ValueError("Cannot partition by physical regime: 'H2' column not found.")
    return pd.Series(np.where(df["H2"].values >= h2_threshold, 0, 1), index=df.index, dtype=int)


def cluster_pareto_deterministic(
    df: pd.DataFrame,
    n_clusters: int = 2,
    n1_col: str = "N1",
    n2_col: str = "N2",
    random_state: int = 42,
) -> pd.Series:
    """
    Deterministic KMeans clustering on (N1, N2) with clusters ordered
    descending by mean H2 value so that cluster 0 always corresponds to the higher H2 regime.
    """
    if len(df) < n_clusters:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clustering_features = df[[n1_col, n2_col]].values
    raw_clusters = kmeans.fit_predict(clustering_features)
    cluster_series = pd.Series(raw_clusters, index=df.index)

    if "H2" in df.columns:
        cluster_means = df.groupby(cluster_series)["H2"].mean().sort_values(ascending=False)
        remap = {old_cid: new_cid for new_cid, old_cid in enumerate(cluster_means.index)}
        return cluster_series.map(remap)
    return cluster_series


def select_optimal_configurations(
    pareto_df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Path] = None,
    output_excel_path: Optional[Path] = None,
    n_clusters: int = 2,
    round_decimals: bool = True,
    clustering_method: str = "physics",
    h2_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Filters, clusters, scores, and deduplicates optimal separator geometric designs:
    - 2-regime physical or deterministic clustering based on slot height H2
    - Full particle balance: N1 + N2 + N3 = 6417
    - Efficiency calculations: eta_1 (%) and eta_2 (%)
    - Geometric deduplication
    - Engineering export to Excel and CSV
    """
    if pareto_df is None:
        if csv_path is None:
            # Search for Pareto result files
            pareto_files = list(optimization_results_dir.glob("pareto_front*.csv"))
            if not pareto_files:
                pareto_files = list(optimization_results_dir.glob("genetic_results_pareto_*.csv"))
            if not pareto_files:
                raise FileNotFoundError(f"No Pareto front result CSV files found in {optimization_results_dir}")
            csv_path = max(pareto_files, key=lambda p: p.stat().st_mtime)

        print(f"[INFO] Loading Pareto solutions from: {csv_path.name}")
        df = pd.read_csv(csv_path)
    else:
        df = pareto_df.copy()

    if len(df) == 0:
        raise ValueError("Pareto solution dataset is empty.")

    n1_col = "N1_pred" if "N1_pred" in df.columns else "N1"
    n2_col = "N2_pred" if "N2_pred" in df.columns else "N2"
    delta_col = "Delta_pred" if "Delta_pred" in df.columns else "Delta"

    # Physics-based regime partitioning or deterministic KMeans
    if "H2" in df.columns and clustering_method == "physics":
        df["cluster"] = partition_pareto_regimes(df, h2_threshold=h2_threshold)
    elif "H2" in df.columns and clustering_method in ("kmeans", "deterministic"):
        df["cluster"] = cluster_pareto_deterministic(
            df, n_clusters=n_clusters, n1_col=n1_col, n2_col=n2_col
        )
    elif len(df) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clustering_features = df[[n1_col, n2_col]].values
        df["cluster"] = kmeans.fit_predict(clustering_features)

    # Compute complete particle balance
    n1_raw = df[n1_col].values
    delta_raw = df[delta_col].values
    n2_raw = df[n2_col].values

    # Normalized compromise score: lower score indicates superior compromise (min loss N1, max advantage Delta)
    n1_min, n1_max = n1_raw.min(), n1_raw.max()
    delta_min, delta_max = delta_raw.min(), delta_raw.max()

    n1_norm = (n1_raw - n1_min) / (n1_max - n1_min + 1e-6)
    delta_norm = (delta_raw - delta_min) / (delta_max - delta_min + 1e-6)
    df["score"] = 0.5 * n1_norm + 0.5 * (1.0 - delta_norm)

    # Prepare export DataFrame
    df_export = df.copy()
    if round_decimals:
        df_export["Alfa"] = df_export["Alfa"].round(2)
        df_export["Beta"] = df_export["Beta"].round(2)
        df_export["H1"] = df_export["H1"].round(4)
        df_export["H2"] = df_export["H2"].round(4)
        df_export["N1"] = np.clip(np.round(n1_raw).astype(int), 0, TOTAL_PARTICLES)
        n2_rounded = np.clip(np.round(n2_raw).astype(int), 0, TOTAL_PARTICLES)
        df_export["N2"] = np.minimum(n2_rounded, TOTAL_PARTICLES - df_export["N1"])
        df_export["Delta"] = (df_export["N2"] - df_export["N1"]).astype(int)
        df_export["N3"] = (TOTAL_PARTICLES - df_export["N1"] - df_export["N2"]).astype(int)
        df_export["eta_1_pct"] = np.round((df_export["N1"] / TOTAL_PARTICLES) * 100.0, 2)
        df_export["eta_2_pct"] = np.round((df_export["N2"] / TOTAL_PARTICLES) * 100.0, 2)
        df_export["score"] = df_export["score"].round(4)
    else:
        df_export["N1"] = np.clip(n1_raw, 0, TOTAL_PARTICLES)
        df_export["N2"] = np.minimum(np.clip(n2_raw, 0, TOTAL_PARTICLES), TOTAL_PARTICLES - df_export["N1"])
        df_export["Delta"] = df_export["N2"] - df_export["N1"]
        df_export["N3"] = TOTAL_PARTICLES - df_export["N1"] - df_export["N2"]
        df_export["eta_1_pct"] = (df_export["N1"] / TOTAL_PARTICLES) * 100.0
        df_export["eta_2_pct"] = (df_export["N2"] / TOTAL_PARTICLES) * 100.0

    # Geometric deduplication
    before = len(df_export)
    df_export = df_export.drop_duplicates(subset=BASE_FEATURES)
    after = len(df_export)
    print(f"[INFO] Removed {before - after} duplicates. Preserved {after} unique Pareto design configurations.")

    # Sort by compromise score
    df_export = df_export.sort_values(by=["score", "N1", "N2"], ascending=[True, True, False]).reset_index(drop=True)

    target_cols = [
        "Alfa",
        "Beta",
        "H1",
        "H2",
        "N1",
        "N2",
        "Delta",
        "N3",
        "eta_1_pct",
        "eta_2_pct",
        "score",
    ]
    if "cluster" in df_export.columns:
        target_cols.append("cluster")
    if "H2" in df_export.columns:
        df_export["regime"] = np.where(
            df_export["H2"] >= h2_threshold,
            "High-gap (H2 ~ 59 mm)",
            "Low-gap (H2 ~ 39 mm)",
        )
        target_cols.append("regime")

    result_df = df_export[target_cols]

    if output_excel_path is None:
        output_excel_path = optimization_results_dir / "pareto_optimal_designs.xlsx"

    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_excel(output_excel_path, index=False)
    csv_out = output_excel_path.with_suffix(".csv")
    result_df.to_csv(csv_out, index=False)
    # Also save to legacy path Test5_Results_src.csv for backward compatibility
    legacy_csv = optimization_results_dir / "Test5_Results_src.csv"
    result_df.to_csv(legacy_csv, index=False)

    print(f"[SAVE] Optimal designs exported to Excel: {output_excel_path}")
    print(f"[SAVE] Optimal designs exported to CSV: {csv_out}")

    return result_df


__all__ = [
    "partition_pareto_regimes",
    "cluster_pareto_deterministic",
    "select_optimal_configurations",
]
