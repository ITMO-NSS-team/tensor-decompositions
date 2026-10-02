"""Convert MLflow histories without multiplying rows at repeated steps."""


class MlflowMetricsPandasConverter:
    def __init__(self, mlFlowClient):
        self.mlFlowClient = mlFlowClient

    def _runs(self, experiment_id, filter_string):
        token, seen = None, set()
        while True:
            options = {"filter_string": filter_string}
            if token is not None:
                options["page_token"] = token
            page = self.mlFlowClient.search_runs([str(experiment_id)], **options)
            yield from page
            token = getattr(page, "token", None)
            if not token:
                break
            if token in seen:
                raise ValueError("MLflow repeated a pagination token")
            seen.add(token)

    @staticmethod
    def _history_by_step(history):
        latest = {}
        for order, metric in enumerate(history):
            candidate = (getattr(metric, "timestamp", 0), order, metric.value)
            if metric.step not in latest or candidate[:2] >= latest[metric.step][:2]:
                latest[metric.step] = candidate
        return {step: row[2] for step, row in latest.items()}

    def get_metrics_from_run(self, run_id, main_metric="loss", include_parameters=None):
        import pandas as pd
        include_parameters = list(include_parameters or ())
        run = self.mlFlowClient.get_run(run_id)
        names = list(dict.fromkeys([main_metric, *run.data.metrics]))
        if set(include_parameters) & {"step", *names}:
            raise ValueError("parameter names must not collide with metric columns")
        missing = set(include_parameters) - run.data.params.keys()
        if missing:
            raise ValueError(f"missing run parameters: {sorted(missing)}")
        histories = {name: self._history_by_step(self.mlFlowClient.get_metric_history(run_id, name))
                     for name in names}
        rows = [{"step": step, **{name: histories[name].get(step) for name in names},
                 **{name: run.data.params[name] for name in include_parameters}}
                for step in sorted(histories[main_metric])]
        return pd.DataFrame(rows, columns=["step", *names, *include_parameters])

    def get_metrics_from_all_experiment_runs(self, experiment_id, filter_string="",
                                             include_parameters=None, main_metric="loss"):
        import pandas as pd
        frames = []
        for run in self._runs(experiment_id, filter_string):
            frame = self.get_metrics_from_run(run.info.run_id, main_metric, include_parameters)
            frame["run_id"], frame["run_name"] = run.info.run_id, run.info.run_name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["step", main_metric, *(include_parameters or ()), "run_id", "run_name"])

    def get_only_last_metrics_from_all_experiment_runs(self, experiment_id, filter_string=""):
        import pandas as pd
        rows = [{"run_id": run.info.run_id, "run_name": run.info.run_name, **run.data.metrics}
                for run in self._runs(experiment_id, filter_string)]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["run_id", "run_name"])
