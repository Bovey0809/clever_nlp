# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
import torch
from transformers.data.data_collator import DataCollatorWithPadding
from dataset import build_dataset
from model import DictNet

checkpoint = "./recnn-10epoch.pt"

model = torch.load(checkpoint)

# %%
dataset, tokenizer = build_dataset((1.3, 1.6))
val_dataset = dataset.shuffle(seed=42)['train'].select(range(100))
# %%
collate_function = DataCollatorWithPadding(tokenizer)
dataloader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=1,
                                         collate_fn=collate_function)

# %%
device = 'cuda:1'
model.to(device)
model.eval()
res = {}
for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    pred_embed = model(**batch)['pred_embed']
    res[int(batch['word_ids'].item())]=pred_embed.detach().cpu()
res
# %%
# calculate embeddings
bert_norm = model.embedding_weight[list(res.keys())].norm(dim=1)
# %%
recnn_norm = torch.cat(list(res.values())).norm(dim=1)
# %%
sum(bert_norm > recnn_norm) / 100
# %%
