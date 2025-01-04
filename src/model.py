from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

from src.config import Config
from src.utils import load_object
from src.metrics import compute_metrics


def create_trainer(config: Config, train_data, eval_data):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_kwargs['model_name'],
        torch_dtype=config.model_kwargs['fmt'],
        num_labels=len(config.data_config.labels),
        label2id={label: idx for idx, label in enumerate(config.data_config.labels)},
        id2label={idx: label for idx, label in enumerate(config.data_config.labels)},
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_kwargs['model_name'])

    if config.model_kwargs['freeze']:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for param in model.bert.encoder.layer[:int(config.model_kwargs['freeze'] * len(model.bert.encoder.layer) / 100)].parameters():
            param.requires_grad = False

#    optimizer = load_object(config.optimizer)(
#        filter(lambda p: p.requires_grad, model.parameters()),
#        **config.optimizer_kwargs,
#    )

#    scheduler = load_object(config.scheduler)(optimizer, **config.scheduler_kwargs)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=10,
        logging_dir='logs',
        per_device_train_batch_size=config.data_config.batch_size,
        per_device_eval_batch_size=config.data_config.batch_size,
        gradient_accumulation_steps=config.model_kwargs['gradient_accumulation_steps'],
        max_grad_norm=config.model_kwargs['max_grad_norm'],
        learning_rate=config.optimizer_kwargs['lr'],
        # weight_decay=config.optimizer_kwargs['weight_decay'],
        # optim_args='weight_decay=5e-8',
        num_train_epochs=config.n_epochs,
        dataloader_num_workers=config.data_config.n_workers,
        load_best_model_at_end=True,
        metric_for_best_model=config.monitor_metric,
        report_to='clearml',
        deepspeed=config.deepspeed_config,
    )

    return Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # optimizers=[optimizer, scheduler],
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )
