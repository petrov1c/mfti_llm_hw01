.PHONY: train
train:
	PYTHONPATH=. python src/train.py config/config.yml

.PHONY: lint
lint:
	flake8 src/*.py