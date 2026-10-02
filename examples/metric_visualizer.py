"""Plot converted MLflow histories with arbitrary DataFrame indices."""


class MetricVisualizer:
    @staticmethod
    def plot_metric_by_run(df, target="loss", step_col="step", run_id_col="run_id",
                           legend_col="run_name", figsize=(12, 7), show=True):
        import pandas as pd
        import matplotlib.pyplot as plt
        required = {run_id_col, step_col, target, legend_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing plot columns: {sorted(missing)}")
        plot_df = df[list(dict.fromkeys([run_id_col, step_col, target, legend_col]))].copy()
        plot_df[run_id_col] = plot_df[run_id_col].astype(str)
        plot_df[step_col] = pd.to_numeric(plot_df[step_col], errors="coerce")
        plot_df[target] = pd.to_numeric(plot_df[target], errors="coerce")
        plot_df = plot_df.dropna(subset=[step_col, target]).sort_values(step_col, kind="stable")
        fig, ax = plt.subplots(figsize=figsize)
        for _, group in plot_df.groupby(run_id_col, sort=False):
            ax.plot(group[step_col], group[target], label=str(group[legend_col].iloc[0]), linewidth=1.5)
        ax.set_xlabel(step_col)
        ax.set_ylabel(target)
        ax.set_title(f"{target} vs {step_col}")
        ax.grid(True, alpha=0.3)
        if not plot_df.empty:
            ax.legend(title=legend_col, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax
