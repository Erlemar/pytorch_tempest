import argparse
import csv
from typing import Dict, Optional, Union, Any
import json
from pytorch_lightning.loggers import LightningLoggerBase


class CsvLogger(LightningLoggerBase):
    @property
    def experiment(self) -> Any:
        pass

    def log_hyperparams(self, params: argparse.Namespace):
        pass

    @property
    def version(self) -> Union[int, str]:
        pass

    def __init__(self, csv_path: str = 'csv_log.csv'):
        # TODO: add folder to filename
        super().__init__()
        self.csv_path = csv_path

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        # print('metrics', metrics)
        # TODO: check empty line?
        if step == 0:
            header = ['step'] + list(metrics.keys())
            with open(self.csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        fields = [step] + list(metrics.values())
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    @property
    def name(self):
        return 'CsvLogger'


class JsonLogger(LightningLoggerBase):
    @property
    def experiment(self) -> Any:
        pass

    def log_hyperparams(self, params: argparse.Namespace):
        pass

    @property
    def version(self) -> Union[int, str]:
        pass

    def __init__(self, json_path: str = 'json_log.json'):
        super().__init__()
        self.json_path = json_path

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        metrics.update({'step': step})
        with open(self.json_path, 'a') as f:
            json.dump(metrics, f)

    @property
    def name(self):
        return 'JsonLogger'
