import comet_ml
from hydra.experimental import initialize, compose
import pandas as pd
import argparse
from omegaconf import DictConfig


def show_scores(cfg: DictConfig, metric: str = 'main_score') -> None:
    comet_api = comet_ml.api.API(cfg.private.comet_api)

    experiments = comet_api.get(f'{cfg.general.workspace}/{cfg.general.project_name}')

    experiment_results = []
    for experiment in experiments:
        scores = experiment.get_metrics('main_score')
        if len(scores) > 10:
            best_score = experiment.get_metrics_summary(metric)['valueMin']
            experiment_results.append((experiment.name, best_score))

    scores = pd.DataFrame(experiment_results, columns=['id', 'score'])
    scores = scores.sort_values('score')
    print(scores.head(10))
    scores.to_csv('saved_objects/scores.csv', index=False)


if __name__ == '__main__':
    """
    You can use this script to see the overview of experiments, if you saved them using comet api
    """
    parser = argparse.ArgumentParser(description='See experiment results for M5')
    parser.add_argument('--config_dir', help='main config dir', type=str, default='conf/')
    parser.add_argument('--main_config', help='main config', type=str, default='config.yaml')
    parser.add_argument('--metric', help='main config', type=str, default='main_score')
    args = parser.parse_args()

    initialize(config_dir=args.config_dir, strict=True)

    cfg = compose(config_file=args.main_config)

    show_scores(cfg, args.metric)
