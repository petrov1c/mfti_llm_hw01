from transformers import AutoTokenizer
from datasets import load_dataset


def glue_dataset(model_name):
    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    glue = load_dataset("glue", "cola", trust_remote_code=True)
    dataset = glue.map(preprocess_function, batched=True)
    return dataset['train'], dataset['validation']

################################################
#   Загрузка датасетов для QA

def prepare_squad(tokenizer):
    def preprocess(examples):
        max_length = 512  # Максимальная длина элемента (вопрос и контекст)
        doc_stride = 128  # Требуется разрешенное перекрытие между двумя частями контекста при его разделении.
        pad_on_right = tokenizer.padding_side == "right"

        examples["question"] = [q.lstrip() for q in examples["question"]]

        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # Мы будем помечать невозможные ответы индексом токена CLS.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Возьмите последовательность, соответствующую этому примеру (чтобы знать, каков контекст и в чем вопрос).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

#    def preprocess(example):
#        # Tokenize question and context
#        encoding = tokenizer(example['question'], example['context'], truncation=True)
#        # Encode input ids and attention masks
#        input_ids = encoding['input_ids']
#        attention_mask = encoding['attention_mask']
#        # Convert answer positions to start and end token indices
#        start_token = input_ids.index(tokenizer.sep_token_id) + 1
#        end_token = len(input_ids) - 1
#        start_index = example['answers']['answer_start'][0]
#        end_index = start_index + len(example['answers']['text'][0])
#        # If answer is out of span, use default values
#        if start_index < start_token or end_index > end_token:
#            start_index = end_index = 0
#        # Return preprocessed example
#        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'start_index': start_index,
#                'end_index': end_index}

    dataset = load_dataset("squad")
    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

    return dataset['train'], dataset['validation']
