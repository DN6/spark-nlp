try:
    import comet_ml
except AttributeError:
    # Python 3.6
    comet_ml = None
except ModuleNotFoundError:
    # Python 3.7+
    comet_ml = None

from typing import List

import threading
import time
import os


class CometLogger:
    def __init__(
        self,
        workspace=None,
        project_name=None,
        comet_mode=None,
        experiment_id=None,
        tags=None,
        **experiment_kwargs,
    ):
        if comet_ml is None:
            raise ImportError(
                "`comet_ml` is not installed. Please install it with `pip install comet-ml`."
            )

        self.comet_mode = comet_mode
        self.workspace = workspace
        self.project_name = project_name
        self.experiment_id = experiment_id
        self.experiment_kwargs = experiment_kwargs

        self.experiment = self._get_experiment(
            self.comet_mode,
            self.workspace,
            self.project_name,
            self.experiment_id,
            **self.experiment_kwargs,
        )
        self.experiment.log_other("Created from", "SparkNLP")
        if tags is not None:
            self.experiment.add_tags(tags)

    def _get_experiment(
        self,
        mode,
        workspace=None,
        project_name=None,
        experiment_id=None,
        **experiment_kwargs,
    ):
        if mode == "offline":
            if experiment_id is not None:
                return comet_ml.ExistingOfflineExperiment(
                    previous_experiment=experiment_id,
                    workspace=workspace,
                    project_name=project_name,
                    **experiment_kwargs,
                )

            return comet_ml.OfflineExperiment(
                workspace=workspace,
                project_name=project_name,
                **experiment_kwargs,
            )

        else:
            if experiment_id is not None:
                return comet_ml.ExistingExperiment(
                    previous_experiment=experiment_id,
                    workspace=workspace,
                    project_name=project_name,
                    **experiment_kwargs,
                )

            return comet_ml.Experiment(
                workspace=workspace,
                project_name=project_name,
                **experiment_kwargs,
            )

    def log_pipeline_parameters(self, pipeline, stages=None):
        """Iterate over the different stages in a pyspark PipelineModel object

        Args:
            pipeline (pyspark.ml.PipelineModel): PipelineModel object
            stages (List[str], optional): Names of the Pipeline Stages to log. Defaults to None.
        """
        self.experiment.log_other("pipeline_uid", pipeline.uid)
        if stages is None:
            stages = [s.name for s in pipeline.stages]

        for stage in pipeline.stages:
            if stage.name not in stages:
                continue

            params = stage.extractParamMap()
            for param, param_value in params.items():
                self.experiment.log_parameter(f"{stage.name}-{param.name}", param_value)

    def log_visualization(self, html, name="viz.html"):
        self.log_asset_data(html, name)

    def log_metrics(self, metrics, step=None, epoch=None):
        self.experiment.log_metrics(metrics, step=step, epoch=epoch)

    def log_parameters(self, parameters, step=None):
        self.experiment.log_parameters(parameters, step=step)

    def log_completed_run(self, log_file_path):
        """Log training metrics after a run has completed

        Args:
            log_file_path (str): Path to log file containing training metrics
        """
        with open(log_file_path, "r") as f:
            stats = f.read().splitlines()

        self._parse_log_entry(stats)
        self.experiment.log_other("log_file_path", log_file_path)

    def log_asset(self, asset_path, metadata=None, step=None):
        self.experiment.log_asset(asset_path, metadata=metadata, step=step)

    def log_asset_data(self, asset, name, overwrite=False, metadata=None, step=None):
        self.experiment.log_asset_data(
            asset, name, overwrite=overwrite, metadata=metadata, step=step
        )

    def monitor(self, logdir, model, interval=10):
        self.experiment.log_other("model_uid", model.uid)
        self.thread = threading.Thread(
            target=self._monitor_log_file,
            args=(
                os.path.join(logdir, f"{model.uid}.log"),
                interval,
            ),
        )
        self.thread.start()

    def _file_watcher(self, filename, interval):
        """Generator that yields lines from the model log file

        Args:
            filename (str): Path to model log file
            interval (int): Time (seconds) to wait in
                between checking for file updates

        Yields:
            str: A single line from the file
        """
        fp = open(filename)

        line = ""
        while True:
            try:
                partial_line = fp.readline()
                if partial_line is not None:
                    line += partial_line
                    if line.endswith("\n"):
                        yield line
                        line = ""

                elif interval:
                    time.sleep(interval)

            except Exception as e:
                break

        fp.close()

    def _monitor_log_file(self, filename, interval):
        # Wait for file to be created:
        while not os.path.exists(filename):
            time.sleep(interval)

        watcher = self._file_watcher(filename, interval)
        for line in watcher:
            lines = line.split("\n")
            self._parse_log_entry(lines)

    def _convert_log_entry_to_dict(self, log_entries):
        """
        Turn a line in the metric log into a dictionary
        """
        output_dict = {}
        for entry in log_entries:
            key, value = entry.strip(" ").split(":")
            output_dict[key] = float(value)

        return output_dict

    def _parse_run_metrics(self, parts):
        epoch_str, ratio = parts[0].split(" ", 1)
        epoch, total = ratio.split("/", 1)

        metrics = parts[2:]
        formatted_metrics = self._convert_log_entry_to_dict(metrics)

        return formatted_metrics, epoch

    def _parse_run_parameters(self, parts):
        parameters = parts[2:]
        formatted_parameters = self._convert_log_entry_to_dict(parameters)
        return formatted_parameters

    def _parse_log_entry(self, lines):
        for line in lines:
            parts = line.split("-")
            if line.startswith("Training started"):
                parameters = self._parse_run_parameters(parts)
                self.log_parameters(parameters)

            elif line.startswith("Epoch"):
                metrics, epoch = self._parse_run_metrics(parts)
                self.log_metrics(metrics, step=int(epoch), epoch=int(epoch))

    def end(self):
        self.experiment.end()
