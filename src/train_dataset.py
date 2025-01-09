import argparse
import logging
import os
import time
import pandas as pd

from clearml import Task
from transformers import AutoTokenizer

from src.config import Config
from src.datamodule import prepare_squad, prepare_coqa, postprocess_qa_predictions
from src.model import create_trainer_qa
from src.metrics import metric_qa


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
        "coqa": prepare_coqa(tokenizer),
        "squad": prepare_squad(tokenizer),
    }

    results = []
    for name, (train_dataset, eval_dataset, validation_features, datasets) in datasets.items():
        start_time = time.time()

        trainer = create_trainer_qa(config, train_dataset, eval_dataset)
        trainer.train()

        raw_predictions = trainer.predict(validation_features)
        validation_features.set_format(type=validation_features.format["type"],
                                       columns=list(validation_features.features.keys()))

        final_predictions = postprocess_qa_predictions(tokenizer, datasets["validation"], validation_features,
                                                       raw_predictions.predictions)

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]

        metrics = metric_qa.compute(predictions=formatted_predictions, references=references)

        end_time = time.time()
        results.append({
            'name': name,
            'em': metrics['exact_match'],
            'f1': metrics['f1'],
            'training_time': end_time - start_time
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('result/results_dataset.csv', index=False, header=True)


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml(args.config_file)
    train(config)
