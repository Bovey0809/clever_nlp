# %%
from datasets.load import load_dataset

# %%
raw_datasets = load_dataset("code_search_net", "python")

# %%
raw_datasets['train']
# %%
raw_datasets['train'][123456]['whole_func_string']
# %%
raw_datasets['train'][123456]['func_documentation_string']
# %%
training_corpus = (raw_datasets['train'][i, i + 1000]['whole_func_string']
                   for i in range(0, len(raw_datasets['train']), 1000))


# %%
def get_training_corpus():
    return (raw_datasets['train'][i: i + 1000]['whole_func_string']
            for i in range(0, len(raw_datasets['train']), 1000))


# %%
training_corpus = get_training_corpus()
# %%
from transformers import AutoTokenizer
# %%
old_tokenizer: = AutoTokenizer.from_pretrained("gpt2")
# %%

# %%

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
# %%
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

# %%
tokens = tokenizer.tokenize(example)
# %%
tokens
# %%
tokenizer.save_pretrained("code-search-net-tokenizer")
# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))
# %%
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# %%
example_word = "81s"
# %%
tokenizer.tokenize(example_word)
# %%
roberta_tokenizer.tokenize(example_word)
# %%
my_sentence1 = "My name is houbowei."
my_sentence2 = "I like eating hamburger."
# %%
encoding = tokenizer(my_sentence1, my_sentence2)
# %%
start, end = encoding.word_to_chars(3, sequence_index=1)
# %%
my_sentence2[start:end]
# %%
from transformers import pipeline

token_classifier = pipeline("token-classification")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
# %%
from transformers import pipeline

token_classifier = pipeline("token-classification", aggregation_strategy="simple")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
# %%
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)
# %%
import torch

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)
# %%
model.config.id2label
# %%
results = []
tokens = inputs.tokens()

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        results.append(
            {"entity": label, "score": probabilities[idx][pred], "word": tokens[idx]}
        )

print(results)
# %%
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
inputs_with_offsets["offset_mapping"]
# %%
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )

print(results)
# %%
import numpy as np

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]
    if label != "O":
        # Remove the B- or I-
        label = label[2:]
        start, _ = offsets[idx]

        # Grab all the tokens labeled with I-label
        all_scores = []
        while (
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] == f"I-{label}"
        ):
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1

        # The score is the mean of all the scores of the tokens in that grouped entity
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

print(results)
# %%
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)
# %%
from transformers import pipeline

question_answerer = pipeline("question-answering")
context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"
question_answerer(question=question, context=context)
# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(type(tokenizer.backend_tokenizer))
# %%
tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?")
# %%
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
# %%
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
# %%
