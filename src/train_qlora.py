import argparse
import logging
import os
import time
import pandas as pd

from clearml import Task

from src.config import Config
from src.datamodule import glue_dataset
from src.model import create_qtrainer


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):
    task = Task.init(project_name='Disable Logging', task_name=config.task_name, auto_connect_frameworks={'pytorch': False})
    os.environ["WANDB_DISABLED"] = "true"

    train_dataset, eval_dataset = glue_dataset(config.model_kwargs['model_name'])
    config.data_config.labels = train_dataset.features['label'].names.copy()

    results = []
    for use_adapter in [True]:
        start_time = time.time()

        trainer = create_qtrainer(config, use_adapter, train_dataset, eval_dataset)
        trainer.train()
        accuracy = trainer.state.best_metric
        # print(f"accuracy: {trainer.state.best_metric}")
        # print(f"Оценка модели: {trainer.evaluate()}")
        # print(f'freeze: {freeze}, type: {fmt}, model type :{trainer.model.dtype}')
        # print(trainer.state)

        end_time = time.time()
        results.append(
            {
                'use_adapter': use_adapter,
                'accuracy': accuracy,
                'training_time': end_time - start_time
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv('result/results_qlora.csv', index=False, header=True)


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml(args.config_file)
    train(config)
