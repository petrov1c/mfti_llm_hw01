import argparse
import logging
import os
import time
import pandas as pd

from clearml import Task

from src.config import Config
from src.datamodule import prepare_russian_superglue
from src.model import create_trainer


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):
    task = Task.init(project_name='Disable Logging', task_name=config.task_name, auto_connect_frameworks={'pytorch': False})
    os.environ["WANDB_DISABLED"] = "true"

    train_dataset, eval_dataset = prepare_russian_superglue(config.model_kwargs['model_name'])
    config.data_config.labels = train_dataset.features['label'].names.copy()

    # Параметры эксперимента
    layer_freezing = [80, 50, 20, 0]  # Процент замороженных слоев
    # data_formats = ['fp32', 'fp16', 'int8']  # Форматы данных
    data_formats = ['float32', 'bfloat16']  # Форматы данных

    results = []
    for freeze in layer_freezing:
        for fmt in data_formats:
            start_time = time.time()

            config.model_kwargs['freeze'] = freeze
            config.model_kwargs['fmt'] = fmt

            trainer = create_trainer(config, train_dataset, eval_dataset)
            trainer.train()
            accuracy = trainer.state.best_metric
            # print(f'freeze: {freeze}, type: {fmt}, model type :{trainer.model.dtype}')
            # print(trainer.state)

            end_time = time.time()
            results.append({
                'frozen_layers': freeze,
                'data_format': fmt,
                'accuracy': accuracy,
                'training_time': end_time - start_time
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False, header=True)


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml(args.config_file)
    train(config)
