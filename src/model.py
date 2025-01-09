from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.config import Config
from src.metrics import compute_metrics
from src.utils import load_object


def load_model_and_tokenizer_from_config(config):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_kwargs['model_name'],
        torch_dtype=config.model_kwargs['fmt'],
        num_labels=len(config.data_config.labels),
        label2id={label: idx for idx, label in enumerate(config.data_config.labels)},
        id2label={idx: label for idx, label in enumerate(config.data_config.labels)},
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_kwargs['model_name'],
    )
    return tokenizer, model


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

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir='logs',
        per_device_train_batch_size=config.data_config.batch_size,
        per_device_eval_batch_size=config.data_config.batch_size,
        gradient_accumulation_steps=config.model_kwargs['gradient_accumulation_steps'],
        max_grad_norm=config.model_kwargs['max_grad_norm'],
        learning_rate=config.optimizer_kwargs['lr'],
        num_train_epochs=config.n_epochs,
        dataloader_num_workers=config.data_config.n_workers,
        load_best_model_at_end=True,
        metric_for_best_model=config.monitor_metric,
        deepspeed=config.deepspeed_config,
        report_to='clearml',
        disable_tqdm=True,
    )

    return Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )


def create_trainer_with_adapter(config: Config, train_dataset, eval_dataset):
    def add_lora_adapter(model, r=16, alpha=32, dropout=0.1):
        config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type='CAUSAL_LM')
        return get_peft_model(model, config)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_kwargs['model_name'],
        torch_dtype=config.model_kwargs['fmt'],
        num_labels=len(config.data_config.labels),
        label2id={label: idx for idx, label in enumerate(config.data_config.labels)},
        id2label={idx: label for idx, label in enumerate(config.data_config.labels)},
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_kwargs['model_name'])

    model = add_lora_adapter(model, **config.adapter_kwargs)

    optimizer = load_object(config.optimizer)(
        filter(lambda p: p.requires_grad, model.parameters()),
        **config.optimizer_kwargs,
    )

    scheduler = load_object(config.scheduler)(optimizer, **config.scheduler_kwargs)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir='logs',
        per_device_train_batch_size=config.data_config.batch_size,
        per_device_eval_batch_size=config.data_config.batch_size,
        gradient_accumulation_steps=config.model_kwargs['gradient_accumulation_steps'],
        max_grad_norm=config.model_kwargs['max_grad_norm'],
        num_train_epochs=config.n_epochs,
        dataloader_num_workers=config.data_config.n_workers,
        load_best_model_at_end=True,
        metric_for_best_model=config.monitor_metric,
        report_to = 'clearml',
        disable_tqdm = True,
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer


def create_trainer_qa(config: Config, train_data, eval_data):
    model = AutoModelForQuestionAnswering.from_pretrained(
        config.model_kwargs['model_name'],
        torch_dtype=config.model_kwargs['fmt'],
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_kwargs['model_name'])

    if 'freeze' in config.model_kwargs and config.model_kwargs['freeze']:
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
        for param in model.base_model.transformer.layer[:int(config.model_kwargs['freeze'] * len(model.base_model.transformer.layer) / 100)].parameters():
            param.requires_grad = False

    optimizer = load_object(config.optimizer)(
        filter(lambda p: p.requires_grad, model.parameters()),
        **config.optimizer_kwargs,
    )

    scheduler = load_object(config.scheduler)(optimizer, **config.scheduler_kwargs)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        save_strategy='epoch',
        logging_dir='logs',
        per_device_train_batch_size=config.data_config.batch_size,
        per_device_eval_batch_size=config.data_config.batch_size,
        max_grad_norm=config.model_kwargs['max_grad_norm'],
        num_train_epochs=config.n_epochs,
        dataloader_num_workers=config.data_config.n_workers,
        report_to='clearml',
#        disable_tqdm=True,
    )

    return Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        optimizers=(optimizer, scheduler),
    )


def create_qtrainer(config: Config, use_adapter, train_dataset, eval_dataset):
    def add_lora_adapter(model, r=16, alpha=32, dropout=0.1):
        config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type='CAUSAL_LM')
        return get_peft_model(model, config)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=config.model_kwargs['fmt'],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=config.model_kwargs['fmt'],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_kwargs['model_name'],
        quantization_config=bnb_config,
        torch_dtype=config.model_kwargs['fmt'],
        num_labels=len(config.data_config.labels),
        label2id={label: idx for idx, label in enumerate(config.data_config.labels)},
        id2label={idx: label for idx, label in enumerate(config.data_config.labels)},
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_kwargs['model_name'])

    model = prepare_model_for_kbit_training(model)
    if use_adapter:
        model = add_lora_adapter(model, **config.adapter_kwargs)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir='logs',
        per_device_train_batch_size=config.data_config.batch_size,
        per_device_eval_batch_size=config.data_config.batch_size,
        gradient_accumulation_steps=config.model_kwargs['gradient_accumulation_steps'],
        optim=config.optimizer,
        learning_rate=config.optimizer_kwargs['lr'],
        lr_scheduler_type='constant',
        max_grad_norm=config.model_kwargs['max_grad_norm'],
        num_train_epochs=config.n_epochs,
        dataloader_num_workers=config.data_config.n_workers,

        bf16=True,
        tf32=True,
        remove_unused_columns=False,
        warmup_ratio=0.03,

        load_best_model_at_end=True,
        metric_for_best_model=config.monitor_metric,
        report_to = 'clearml',
        disable_tqdm = True,
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer
