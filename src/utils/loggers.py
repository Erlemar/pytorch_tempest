import argparse
import csv
from typing import Dict, Optional, Union, Any

from pytorch_lightning.loggers import LightningLoggerBase

#
# class CsvLogger(LightningLoggerBase):
#     # TODO Check if works
#     @property
#     def experiment(self) -> Any:
#         pass
#
#     def log_hyperparams(self, params: argparse.Namespace) -> Any:
#         pass
#
#     @property
#     def version(self) -> Union[int, str]:
#         pass
#
#     def __init__(self, csv_path: str = 'csv_log.csv'):
#         # TODO: add folder to filename
#         super().__init__()
#         self.csv_path = csv_path
#
#     def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> Any:
#         if step == 0:
#             header = ['step'] + list(metrics.keys())
#             with open(self.csv_path, 'w') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(header)
#         if step:
#             fields = [float(step)] + list(metrics.values())
#         else:
#             fields = list(metrics.values())
#         with open(self.csv_path, 'a') as f:
#             writer = csv.writer(f)
#             writer.writerow(fields)
#
#     @property
#     def name(self):
#         return 'CsvLogger'
#
#
# class PrintLogger(LightningLoggerBase):
#     # TODO Check if works
#     @property
#     def experiment(self) -> Any:
#         pass
#
#     def log_hyperparams(self, params: argparse.Namespace) -> Any:
#         pass
#
#     @property
#     def version(self) -> Union[int, str]:
#         pass
#
#     def __init__(self, csv_path: str = 'csv_log.csv'):
#         # TODO: add folder to filename
#         super().__init__()
#         self.csv_path = csv_path
#
#     def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> Any:
#         if step == 0:
#             header = ['step'] + list(metrics.keys())
#             print(header)
#         if step:
#             fields = [float(step)] + list(metrics.values())
#         else:
#             fields = list(metrics.values())
#         print(fields)
#
#     @property
#     def name(self):
#         return 'PrintLogger'
