# %%
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from transformers.data import DataCollatorWithPadding
from transformers.models.auto.tokenization_auto import AutoTokenizer

from dataset import build_dataset, word_dict
from model import DictNet
from evaluation import norm_recall, cosine_recall

weight_path = '/diskb/houbowei/ray_results/train_2022-01-05_20-09-55/train_64e33_00107_107_batch_size=8,epochs=100,lr=0.001_2022-01-05_20-16-11/checkpoint_000098/checkpoint'
state_dict = torch.load(weight_path)

model = DictNet()
model.load_state_dict(state_dict=state_dict[0])

eval_dataset, tokenizer = build_dataset((0, 2))

# %%
xp = "a young person who likes this music, wears mainly black clothes, and is often nervous, worried, and unhappy"


def demo_definition(definition):
    model.eval()
    # %%
    inputs = tokenizer(definition, return_tensors='pt')
    inputs['word_ids'] = torch.tensor(111)
    # %%
    pred_embed = model(**inputs)
    # %%
    indexes = norm_recall(pred_embed, model.embedding_weight)
    print(tokenizer.convert_ids_to_tokens(indexes[0]))

    indexes = cosine_recall(pred_embed, model.embedding_weight)
    print(tokenizer.convert_ids_to_tokens(indexes[0]))


print(tokenizer.tokenize('Kočka'))

for i in [xp]:
    demo_definition(i)
    print("\n")