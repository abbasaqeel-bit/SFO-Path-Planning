"""Tables and figures for a completed benchmark batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


DEFAULT_METRICS = (
    "path_length",
    "execution_time_s",
    "cpu_time_s",
    "peak_memory_mb",
    "total_turning_angle_rad",
    "minimum_clearance",
    "path_efficiency",
    "expanded_nodes",
    "generated_nodes",
    "search_iterations",
    "fitness_evaluations",
)


class EvaluationReporter:
    """Generate aggregate CSV/Excel tables and benchmark figures."""

    def __init__(
        self,
        experiment_root: Path,
        records: list[dict[str, Any]],
    ) -> None:
        self.experiment_root = Path(experiment_root)
        self.records = records
        self.tables_dir = self.experiment_root / "tables"
        self.figures_dir = self.experiment_root / "figures"
        self.convergence_dir = self.figures_dir / "convergence"
        self.boxplots_dir = self.figures_dir / "boxplots"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.convergence_dir.mkdir(parents=True, exist_ok=True)
        self.boxplots_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self) -> None:
        if not self.records:
            return
        frame = pd.DataFrame(self.records)
        self._normalise(frame)
        self._generate_tables(frame)
        self._generate_boxplots(frame)
        self._generate_convergence_figures()

    @staticmethod
    def _normalise(frame: pd.DataFrame) -> None:
        if "success" in frame.columns:
            frame["success"] = frame["success"].astype(bool)
        for column in DEFAULT_METRICS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

    def _generate_tables(self, frame: pd.DataFrame) -> None:
        group_keys = ["map_id", "algorithm"]
        metric_columns = [
            column for column in DEFAULT_METRICS if column in frame.columns
        ]
        successful = frame[frame["success"]].copy()
        unsuccessful = frame[~frame["success"]].copy()

        success = (
            frame.groupby(group_keys, dropna=False)
            .agg(
                total_runs=("success", "size"),
                successful_runs=("success", "sum"),
            )
            .reset_index()
        )
        success["success_rate_percent"] = (
            100.0 * success["successful_runs"] / success["total_runs"]
        )

        if successful.empty:
            descriptive = pd.DataFrame(columns=group_keys)
            publication = success.copy()
        else:
            descriptive = (
                successful.groupby(group_keys)[metric_columns]
                .agg(["mean", "std", "median", "min", "max"])
                .reset_index()
            )
            descriptive.columns = [
                "_".join(str(part) for part in column if str(part))
                if isinstance(column, tuple)
                else str(column)
                for column in descriptive.columns
            ]

            selected = [
                column
                for column in (
                    "path_length",
                    "execution_time_s",
                    "total_turning_angle_rad",
                    "minimum_clearance",
                    "path_efficiency",
                    "generated_nodes",
                    "fitness_evaluations",
                )
                if column in successful.columns
            ]
            aggregations: dict[str, tuple[str, str]] = {}
            for metric in selected:
                aggregations[f"{metric}_mean"] = (metric, "mean")
                aggregations[f"{metric}_std"] = (metric, "std")
            publication = (
                successful.groupby(group_keys)
                .agg(**aggregations)
                .reset_index()
                .merge(success, on=group_keys, how="outer")
            )

        ranks = self._mean_ranks(successful, metric_columns)
        friedman, pairwise = self._statistical_tests(successful)

        frame.to_csv(
            self.tables_dir / "all_runs.csv",
            index=False,
            encoding="utf-8-sig",
        )
        unsuccessful.to_csv(
            self.tables_dir / "failed_and_partial_runs.csv",
            index=False,
            encoding="utf-8-sig",
        )
        success.to_csv(
            self.tables_dir / "success_rates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        descriptive.to_csv(
            self.tables_dir / "descriptive_statistics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        publication.to_csv(
            self.tables_dir / "summary_table.csv",
            index=False,
            encoding="utf-8-sig",
        )
        ranks.to_csv(
            self.tables_dir / "mean_ranks.csv",
            index=False,
            encoding="utf-8-sig",
        )
        friedman.to_csv(
            self.tables_dir / "friedman_tests.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pairwise.to_csv(
            self.tables_dir / "pairwise_wilcoxon_holm.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with pd.ExcelWriter(
            self.tables_dir / "benchmark_results.xlsx",
            engine="openpyxl",
        ) as writer:
            publication.to_excel(
                writer,
                sheet_name="Summary Table",
                index=False,
            )
            descriptive.to_excel(
                writer,
                sheet_name="Descriptive Statistics",
                index=False,
            )
            success.to_excel(
                writer,
                sheet_name="Success Rates",
                index=False,
            )
            ranks.to_excel(writer, sheet_name="Mean Ranks", index=False)
            friedman.to_excel(
                writer,
                sheet_name="Friedman Tests",
                index=False,
            )
            pairwise.to_excel(
                writer,
                sheet_name="Wilcoxon Holm",
                index=False,
            )
            frame.to_excel(writer, sheet_name="Raw Results", index=False)
            unsuccessful.to_excel(
                writer,
                sheet_name="Failed Partial Runs",
                index=False,
            )

    @staticmethod
    def _mean_ranks(
        successful: pd.DataFrame,
        metric_columns: list[str],
    ) -> pd.DataFrame:
        """Rank path quality conditional on successful runs."""
        if successful.empty:
            return pd.DataFrame(columns=["map_id", "algorithm"])
        rank_metrics = [
            column
            for column in (
                "path_length",
                "execution_time_s",
                "total_turning_angle_rad",
            )
            if column in metric_columns
        ]
        rows: list[pd.DataFrame] = []
        for map_id, subset in successful.groupby("map_id"):
            means = subset.groupby("algorithm")[rank_metrics].mean()
            ranked = means.rank(axis=0, method="average", ascending=True)
            ranked["overall_mean_rank"] = ranked.mean(axis=1)
            ranked.insert(0, "map_id", map_id)
            rows.append(ranked.reset_index())
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    @staticmethod
    def _holm_adjust(p_values: list[float]) -> list[float]:
        count = len(p_values)
        if count == 0:
            return []
        order = np.argsort(np.asarray(p_values, dtype=float))
        adjusted = np.empty(count, dtype=float)
        running = 0.0
        for rank, index in enumerate(order):
            value = min(1.0, (count - rank) * float(p_values[index]))
            running = max(running, value)
            adjusted[index] = running
        return adjusted.tolist()

    @classmethod
    def _statistical_tests(
        cls,
        successful: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Use only complete paired seed blocks among successful runs.

        These path-quality tests are conditional on success. The separate
        success-rate table must be interpreted first.
        """
        required_pairing_columns = {"map_id", "algorithm", "run_number"}
        if successful.empty or not required_pairing_columns.issubset(
            successful.columns
        ):
            return pd.DataFrame(), pd.DataFrame()

        metrics = [
            column
            for column in (
                "path_length",
                "execution_time_s",
                "total_turning_angle_rad",
                "minimum_clearance",
                "path_efficiency",
            )
            if column in successful.columns
        ]
        friedman_rows: list[dict[str, Any]] = []
        pairwise_rows: list[dict[str, Any]] = []

        for map_id, map_frame in successful.groupby("map_id"):
            for metric in metrics:
                raw_pivot = map_frame.pivot_table(
                    index="run_number",
                    columns="algorithm",
                    values=metric,
                    aggfunc="first",
                )
                pivot = raw_pivot.dropna(axis=0, how="any")

                if pivot.shape[0] >= 3 and pivot.shape[1] >= 3:
                    samples = [
                        pivot[column].to_numpy(dtype=float)
                        for column in pivot.columns
                    ]
                    statistic, p_value = friedmanchisquare(*samples)
                    friedman_rows.append(
                        {
                            "map_id": map_id,
                            "metric": metric,
                            "complete_blocks": int(pivot.shape[0]),
                            "algorithms": int(pivot.shape[1]),
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                        }
                    )

                proposed = "three_phase_incremental_sfo"
                if proposed not in raw_pivot.columns:
                    continue
                family_rows: list[dict[str, Any]] = []
                raw_p_values: list[float] = []
                for comparator in raw_pivot.columns:
                    if comparator == proposed:
                        continue
                    paired = raw_pivot[[proposed, comparator]].dropna()
                    if len(paired) < 3:
                        continue
                    first = paired[proposed].to_numpy(dtype=float)
                    second = paired[comparator].to_numpy(dtype=float)
                    if np.allclose(first, second):
                        statistic, p_value = 0.0, 1.0
                    else:
                        statistic, p_value = wilcoxon(
                            first,
                            second,
                            alternative="two-sided",
                            zero_method="wilcox",
                        )
                    family_rows.append(
                        {
                            "map_id": map_id,
                            "metric": metric,
                            "proposed": proposed,
                            "comparator": comparator,
                            "paired_blocks": int(len(paired)),
                            "statistic": float(statistic),
                            "p_value_raw": float(p_value),
                        }
                    )
                    raw_p_values.append(float(p_value))

                adjusted = cls._holm_adjust(raw_p_values)
                for row, adjusted_p in zip(
                    family_rows,
                    adjusted,
                    strict=True,
                ):
                    row["p_value_holm"] = adjusted_p
                    row["significant_0_05"] = adjusted_p < 0.05
                    pairwise_rows.append(row)

        return pd.DataFrame(friedman_rows), pd.DataFrame(pairwise_rows)

    def _generate_boxplots(self, frame: pd.DataFrame) -> None:
        successful = frame[frame["success"]]
        metrics = [
            column
            for column in (
                "path_length",
                "execution_time_s",
                "total_turning_angle_rad",
                "minimum_clearance",
                "path_efficiency",
            )
            if column in frame.columns
        ]
        for map_id, all_map_runs in frame.groupby("map_id"):
            all_labels = sorted(
                all_map_runs["algorithm"].astype(str).unique()
            )
            valid_counts = (
                all_map_runs.groupby("algorithm")["success"].sum().to_dict()
            )
            total_counts = (
                all_map_runs.groupby("algorithm")["success"].size().to_dict()
            )
            for metric in metrics:
                map_frame = (
                    all_map_runs
                    if metric == "execution_time_s"
                    else successful[successful["map_id"] == map_id]
                )
                data = []
                positions = []
                for position, algorithm in enumerate(all_labels, start=1):
                    algorithm_frame = map_frame[
                        map_frame["algorithm"].astype(str) == algorithm
                    ]
                    values = (
                        algorithm_frame[metric]
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                    if values.size:
                        data.append(values)
                        positions.append(position)
                figure_height = max(6.0, 0.55 * len(all_labels) + 2.0)
                figure, axis = plt.subplots(figsize=(11, figure_height))
                if data:
                    axis.boxplot(
                        data,
                        positions=positions,
                        vert=False,
                        showmeans=True,
                    )
                axis.set_yticks(range(1, len(all_labels) + 1))
                axis.set_yticklabels(all_labels)
                axis.set_ylabel("Algorithm")
                axis.set_xlabel(metric.replace("_", " ").title())
                axis.set_title(
                    f"{metric.replace('_', ' ').title()} — {map_id}"
                )
                for position, algorithm in enumerate(all_labels, start=1):
                    valid = int(valid_counts.get(algorithm, 0))
                    total = int(total_counts.get(algorithm, 0))
                    if metric != "execution_time_s" and valid == 0:
                        axis.annotate(
                            f"No valid path (0/{total})",
                            xy=(0.01, position),
                            xycoords=("axes fraction", "data"),
                            color="#d62728",
                            fontsize=8,
                            va="center",
                            weight="bold",
                        )
                    else:
                        axis.annotate(
                            f"valid {valid}/{total}",
                            xy=(0.99, position),
                            xycoords=("axes fraction", "data"),
                            ha="right",
                            va="center",
                            fontsize=7,
                            color="0.35",
                        )
                axis.grid(axis="x", linewidth=0.4, alpha=0.5)
                figure.tight_layout()
                figure.savefig(
                    self.boxplots_dir / f"{map_id}_{metric}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(figure)

    def _generate_convergence_figures(self) -> None:
        """Generate one native-objective figure per algorithm and map."""
        excluded = {"tables", "figures"}
        for map_dir in [
            path
            for path in self.experiment_root.iterdir()
            if path.is_dir() and path.name not in excluded
        ]:
            for algorithm_dir in map_dir.iterdir():
                if not algorithm_dir.is_dir():
                    continue
                histories: list[np.ndarray] = []
                for run_dir in sorted(algorithm_dir.glob("Run_*")):
                    state_path = run_dir / "planner_state.json"
                    if not state_path.exists():
                        continue
                    try:
                        payload = json.loads(
                            state_path.read_text(encoding="utf-8")
                        )
                        history = np.asarray(
                            payload.get("fitness_history", []),
                            dtype=float,
                        )
                        history = history[np.isfinite(history)]
                    except (
                        OSError,
                        ValueError,
                        TypeError,
                        json.JSONDecodeError,
                    ):
                        continue
                    if history.size:
                        histories.append(history)
                if not histories:
                    continue

                maximum = max(len(history) for history in histories)
                aligned = np.vstack(
                    [
                        np.pad(
                            history,
                            (0, maximum - len(history)),
                            constant_values=history[-1],
                        )
                        for history in histories
                    ]
                )
                mean = np.mean(aligned, axis=0)
                std = np.std(aligned, axis=0)
                x = np.arange(1, maximum + 1)
                algorithm = algorithm_dir.name

                pd.DataFrame(
                    {
                        "iteration": x,
                        "mean_best_native_objective": mean,
                        "std_best_native_objective": std,
                        "runs": aligned.shape[0],
                    }
                ).to_csv(
                    self.convergence_dir
                    / f"{map_dir.name}_{algorithm}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                figure, axis = plt.subplots(figsize=(8.5, 5.5))
                line = axis.plot(
                    x,
                    mean,
                    linewidth=1.8,
                    label=algorithm,
                )[0]
                axis.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    alpha=0.14,
                    color=line.get_color(),
                )
                axis.set_xlabel("Recorded native iteration / evaluation")
                axis.set_ylabel("Best native objective value")
                axis.set_title(
                    f"Native convergence — {algorithm} — {map_dir.name}"
                )
                axis.grid(True, linewidth=0.4, alpha=0.5)
                axis.legend(fontsize=8)
                figure.tight_layout()
                figure.savefig(
                    self.convergence_dir
                    / f"{map_dir.name}_{algorithm}_native_convergence.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(figure)
