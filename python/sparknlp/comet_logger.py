import comet_ml

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
        self.experiment = comet_ml.Experiment()
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
        self.experiment.log_asset_data(html, name)

    def log_metrics(self, metrics, step=None, epoch=None):
        self.experiment.log_metrics(metrics, step=step, epoch=epoch)

    def log_parameters(self, parameters, step=None):
        self.experiment.log_parameters(parameters, step=step)

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

    def log_run(self, model):
        log_file_path = f"./{model.uid}.log"
        self.experiment.log_other("model_uid", model.uid)
        with open(log_file_path, "r") as f:
            stats = f.read().splitlines()
        self._parse_log_entry(stats)

    def __log_watcher(self, filename, interval):
        # Wait for file to be created:
        while not os.path.exists(filename):
            time.sleep(interval)

        # Open file and read lines when ready:
        fp = open(filename)
        while True:
            try:
                position = fp.tell()
                data = fp.read()

                if data.endswith("\n"):
                    lines = data.split("\n")
                    self._parse_log_entry(lines)

                else:
                    if "\n" in data:
                        newline_position = data.rindex("\n")
                        lines = data[:newline_position].split("\n")
                        fp.seek(position + newline_position + 3)
                        self._parse_log_entry(lines)
                    else:
                        fp.seek(position)

                time.sleep(10)
            except Exception as exc:
                print(exc)
                break

        fp.close()

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

        while True:
            try:
                line = fp.readline()
                if line is not None:
                    if line.endswith("\n"):
                        yield line
                elif interval:
                    time.sleep(interval)

            except Exception as e:
                print(e)
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
                self.experiment.log_parameters(parameters)

            elif line.startswith("Epoch"):
                metrics, epoch = self._parse_run_metrics(parts)
                self.log_metrics(metrics, step=int(epoch), epoch=int(epoch))

    def end(self):
        self.experiment.end()
