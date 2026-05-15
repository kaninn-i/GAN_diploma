import os
import json
import time
from datetime import datetime


RUNS_ROOT = "runs"


def ensure_runs_dir():
    os.makedirs(RUNS_ROOT, exist_ok=True)


def create_experiment():
    """
    Создаёт новую папку запуска:
    runs/exp_20260510_194512/
    """
    ensure_runs_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{timestamp}"

    exp_dir = os.path.join(RUNS_ROOT, exp_name)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "dataset"), exist_ok=True)

    return exp_dir


def save_experiment_config(exp_dir, config):
    config_path = os.path.join(exp_dir, "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False
        )


def save_metrics(exp_dir, metrics):
    metrics_path = os.path.join(exp_dir, "metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False
        )


class StageTimer:

    def __init__(self):
        self.data = {}
        self.current_stage = None
        self.start_time = None

    def start(self, stage_name):
        self.current_stage = stage_name
        self.start_time = time.time()

    def stop(self):
        if self.current_stage is None:
            return

        elapsed = time.time() - self.start_time

        self.data[self.current_stage] = round(elapsed, 2)

        self.current_stage = None
        self.start_time = None

    def export(self):
        return self.data