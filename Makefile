.PHONY: train
train:
	PYTHONPATH=. python src/train.py config/config.yml

train_adapter:
	PYTHONPATH=. python src/train_adapter.py config/config_adapter.yml

.PHONY: lint
lint:
	flake8 src/*.py