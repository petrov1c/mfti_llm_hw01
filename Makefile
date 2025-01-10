.PHONY: train
train_freeze:
	PYTHONPATH=. python src/train_freeze.py config/config_freeze.yml

train_lora:
	PYTHONPATH=. python src/train_lora.py config/config_lora.yml

train_qa:
	PYTHONPATH=. python src/train_qa.py config/config_qa.yml

train_qlora:
	PYTHONPATH=. python src/train_qlora.py config/config_qlora.yml

.PHONY: lint
lint:
	flake8 src/*.py