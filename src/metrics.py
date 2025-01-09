import evaluate
import numpy as np

metric = evaluate.load('accuracy')
metric_qa = evaluate.load('squad')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
