# %%
# Train a classifier
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
import torch

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils.dummy_pt_objects import AdamW
from transformers import AdamW
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = ["Merry Christmas", "Hello world!"]

batch = tokenizer(sequences,
                  padding=True,
                  truncation=True,
                  return_tensors='pt')


# %%
batch["labels"] = torch.tensor([1, 1])
# %%
optimier = AdamW(model.parameters())

# %%
loss =model(**batch).loss
# %%
loss.backward()
# %%
optimier.step()
# %%
from datasets import load_dataset
# %%
