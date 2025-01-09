import argparse
import logging
import os
import time
import pandas as pd

from clearml import Task
from transformers import AutoTokenizer

from src.config import Config
from src.datamodule import prepare_squad
from src.model import create_trainer_qa


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config):
    task = Task.init(project_name='Disable Logging', task_name=config.task_name, auto_connect_frameworks={'pytorch': False})
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tokenizer = AutoTokenizer.from_pretrained(config.model_kwargs['model_name'])

    datasets = {
        #    "coqa": load_dataset("coqa"),
        "squad": prepare_squad(tokenizer)
    }

    results = []
    for name, (train_dataset, eval_dataset) in datasets.items():
        start_time = time.time()

        trainer = create_trainer_qa(config, train_dataset, eval_dataset)
        trainer.train()
        accuracy = trainer.state.best_metric
        print(f"accuracy: {trainer.state.best_metric}")
        print(f"Оценка модели: {trainer.evaluate()}")
        print(f'model type :{trainer.model.dtype}')
        print(trainer.state)

        end_time = time.time()
        results.append({
            'name': name,
            # 'data_format': fmt,
            'accuracy': accuracy,
            'training_time': end_time - start_time
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('result/results_dataset.csv', index=False, header=True)


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml(args.config_file)
    train(config)
