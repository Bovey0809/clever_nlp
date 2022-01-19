# %%
# Bert on NER
import re
from datasets.arrow_dataset import concatenate_datasets
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForMaskedLM
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import TrainingArguments

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

imdb_dataset = load_dataset("imdb")

# %%
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")
# %%
# %%
# Create random sample of the unsuperviesed split and verify the label.
unsupervised_dataset = imdb_dataset['unsupervised']

# %%
# Verify the labels in the unsupervised data to be neither 0 or 1.
for i in unsupervised_dataset:
    assert i['label'] != 0 and i['label'] != 1

# %%
for i in imdb_dataset['train']:
    assert i['label'] == 0 or i['label'] == 1
# %%
for i in imdb_dataset['test']:
    assert i['label'] == 0 or i['label'] == 1


# %%
def tokenize_function(examples):
    result = tokenizer(examples['text'])
    if tokenizer.is_fast:
        result['word_ids'] = [
            result.word_ids(i) for i in range(len(result['input_ids']))
        ]
    return result


# %%
tokenized_dataset = imdb_dataset.map(tokenize_function,
                                     batched=True,
                                     remove_columns=['text', 'label'])
# %%
bigbird_tokenizer = AutoTokenizer.from_pretrained(
    "google/bigbird-roberta-base")

# %%
chunk_size = 128


# %%
def group_texts(examples):
    # concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i:i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result
# %%
lm_datasets = tokenized_dataset.map(group_texts, batched=True)
# %%
# %%
from transformers import DataCollatorForLanguageModeling
# %%
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
# %%
samples = [lm_datasets['train'][i] for i in range(2)]
# %%
for sample in samples:
    _ = sample.pop('word_ids')
# %%
for chunk in data_collator(samples)['input_ids']:
    print(f"{tokenizer.decode(chunk)}")
    print(tokenizer.convert_ids_to_tokens(chunk))
# %%
import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return default_data_collator(features)
# %%
samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
    print(tokenizer.convert_ids_to_tokens(chunk))
# %%
train_size = 10_000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
downsampled_dataset
# %%

# %%
batch_size = 64
logging_steps = len(downsampled_dataset['train']) // batch_size
# %%
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
)
# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
)
# %%
import math
eval_results = trainer.evaluate()
print(math.exp(eval_results['eval_loss']))
# %%

trainer.train()
# %%
# %%
import math

eval_results = trainer.evaluate()
print(math.exp(eval_results['eval_loss']))
# %%
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
# %%
downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
# %%
from torch.utils.data import DataLoader
from transformers import default_data_collator

batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(eval_dataset,
                             batch_size=batch_size,
                             collate_fn=default_data_collator)

# %%
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
# %%
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
# %%
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
# %%
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
# %%
from huggingface_hub import get_full_repo_name

model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name
# %%
from huggingface_hub import Repository

output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)
# %%
from tqdm.auto import tqdm
import torch
import math

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
# %%
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask", model="huggingface-course/distilbert-base-uncased-finetuned-imdb"
)
# %%
text = "this is a great [MASK]."
preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
# %%
