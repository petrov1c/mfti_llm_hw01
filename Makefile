.PHONY: train
train:
	PYTHONPATH=. python src/train.py config/config.yml

train_adapter:
	PYTHONPATH=. python src/train_adapter.py config/config_adapter.yml

train_qa:
	PYTHONPATH=. python src/train_qa.py config/config_qa.yml

train_qlora:
	PYTHONPATH=. python src/train_qlora.py config/config_qlora.yml

.PHONY: lint
lint:
	flake8 src/*.py