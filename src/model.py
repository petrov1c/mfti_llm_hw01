from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

from src.metrics import compute_metrics
from src.config import Config
from src.utils import load_object


def create_trainer(config: Config, train_data, eval_data):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_kwargs['model_name'],
        num_labels=len(config.data_config.labels),
        label2id={label: idx for idx, label in enumerate(config.data_config.labels)},
        id2label={idx: label for idx, label in enumerate(config.data_config.labels)},
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_kwargs['model_name'])

    # ToDo добавить заморозку слоев
    optimizer = load_object(config.optimizer)(
        model.parameters(),
        **config.optimizer_kwargs,
    )

    scheduler = load_object(config.scheduler)(optimizer, **config.scheduler_kwargs)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=50,
        per_device_train_batch_size=config.data_config.batch_size,
        per_gpu_eval_batch_size=config.data_config.batch_size,
        num_train_epochs=config.n_epochs,
        load_best_model_at_end=True,
        metric_for_best_model=config.monitor_metric,
    )

    return Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=[optimizer, scheduler],
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )
