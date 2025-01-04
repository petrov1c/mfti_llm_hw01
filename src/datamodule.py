from transformers import AutoTokenizer
from datasets import load_dataset


def prepare_russian_superglue(model_name):
    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    glue = load_dataset("glue", "cola", trust_remote_code=True)
    dataset = glue.map(preprocess_function, batched=True)
    return dataset['train'], dataset['validation']
