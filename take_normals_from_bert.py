#!/usr/bin/env python
# coding: utf-8

# # Tasks
# - [x] 提取模长比较短的词.

# In[1]:

from transformers import AutoModel

# In[2]:

chekpoint = "bert-base-uncased"

# In[3]:

model = AutoModel.from_pretrained(chekpoint)

# In[4]:

model.embeddings

# ## 为什么word embedding的输入是30522? 回答: 是字典的长度

# In[5]:

from transformers import AutoTokenizer

# In[6]:

tokenizer = AutoTokenizer.from_pretrained(chekpoint)

# In[7]:

text = "I'm using bert."

# In[8]:

tokenizer(text)

# In[9]:

len(tokenizer.vocab)

# 词典的个数就是 word_embeddings 的深度.

# In[10]:

inputs = tokenizer(text)

# In[11]:

tokenizer.save_vocabulary("./")

# # 拿出norm的值比较低的看一下.
# 1. 计算所有的vocabulary的norm的值.
# 2. 排序之后查看norm最大的10个对应的单词.

# In[12]:

model.embeddings.state_dict().keys()

# In[13]:

embeddings_weights = model.embeddings.state_dict()['word_embeddings.weight']

# In[14]:

# calculate norms for embeddings.
words_norms = embeddings_weights.norm(dim=1)

# In[47]:

# topk
K = 20
indexes = words_norms.argsort(descending=True)[:K]
# change tokens
tokens = tokenizer.convert_ids_to_tokens(indexes)
print(f"There are {len(tokens)} with longest norms.")
tokenizer.convert_tokens_to_string(tokens)
for token in tokens:
    print(tokenizer.convert_tokens_to_string([token]))

# In[48]:

# topk
K = 20
indexes = words_norms.argsort(descending=False)[:K]
# change tokens
tokens = tokenizer.convert_ids_to_tokens(indexes)
print(f"There are {len(tokens)} with shortest norms.")
tokenizer.convert_tokens_to_string(tokens)
for token in tokens:
    print(tokenizer.convert_tokens_to_string([token]))

# 更长的模长的单词确实都不知道是什么意思, 除了“[CLS]”这个token, 最短的模长的单词都是常见词.
#
# 模长 | 意思
# -- | --
# 长 | 看不懂
# 短 | 常见词

# - [x] TODO: 找到50个低频词汇和字典的交集
# 不需要作这一步, 这一步是循环论证了.

# # 构建训练数据集
# 构建数据集需要在已有的vocabulary里面找到在wordnet对应的单词.
# 1. wordnet 和 bert 的 vocabulary 求交集.
# 2. 求完交集之后的单词列表再对 imdb 数据集筛选.

# In[50]:

from typing import Counter

from datasets import load_dataset


def word_dict() -> dict:
    data_files = "/diskb/houbowei/clever_nlp/core-wordnet.txt"

    def _parse_words(words: list):
        word = words[1]
        word = word.strip("[]")
        explanation = ' '.join(words[2:]).strip()
        return word, explanation

    wordnet = {}
    with open(data_files, 'r') as f:
        text = f.readline()
        while text:
            words = text.split()
            assert len(words) >= 3, f"{words}"
            word, explanation = _parse_words(words[1:])
            if explanation and word:
                wordnet[word] = explanation
            text = f.readline()
        print("DONE")
    return wordnet


imdb_dataset = load_dataset("imdb")

# In[51]:

wordnet = word_dict()

# In[55]:

words_wordnet = list(wordnet.keys())

# In[60]:

words_bert = list(tokenizer.vocab.keys())

# In[70]:

print(f"there are {len(words_wordnet)} in wordnet")

# In[72]:

print(f"there are {len(words_bert)} in bert.")

# In[68]:

common_words = set(words_bert) & set(words_wordnet)

# In[73]:

print(f"there are {len(common_words)} in wordnet and bert")

# - [ ] 由双方共有的单词表来构建训练数据集

# In[76]:

imdb_dataset['train']

# - [ ] 寻找common words里面的低频词汇

# In[95]:

# words -> tokens --> norms
common_words_norms = {}
for word in common_words:
    token = tokenizer.tokenize(word)
    i = tokenizer.convert_tokens_to_ids(token)
    common_words_norms[word] = words_norms[i].numpy()

# 上一步得到了单词表里面的单词的词频, 下一步需要计算下模长最长的单词是啥.

# In[104]:

dict(sorted(common_words_norms.items(), key=lambda item: item[1],
            reverse=True))

# owe确实在wordnet里面, 不过这些单词我都认识呀....不过这样也确实证明了几个事情
# 1. embedding的模长没有取错.
#
# 现在拥有了一个低频(低模)的词典, 并且这个词典里面的词都是bert的语料里面出现过的.
# 现在我需要训练一个网络, 让这些低频词的模长发生变化, 集体变小.

# - [ ] TODO: 利用wordnet的字典做一个recnn的net.
# - [ ] 利用pandas做一个transformer的数据集, 先做train, 里面包含 features 是 [explanation] [word] [embeddings]

# In[140]:

# Select common words from wordnet, common words is the word in bert.
common_words_dict = {k: v for k, v in wordnet.items() if k in common_words}

# In[165]:

import pandas as pd

wordnet_series = pd.DataFrame(
    {
        'word': common_words_dict.keys(),
        'explanation': common_words_dict.values()
    },
    index=None)

# In[168]:

# change series to transformer's dataset
# 1. create txt wordnet
wordnet_series.to_csv("wordnet.csv", index=None)

# In[199]:

wordnet_dataset = load_dataset('csv', data_files="wordnet.csv")

# # 利用数据集训练网络
# - [ ] 构建一个网络

# In[279]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer


class DictNet(nn.Module):
    """Some Information about CleverNLP"""
    def __init__(self, model='bert-base-uncased', device='cuda'):
        super(DictNet, self).__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(model)
        self.embedding_weight = self.bert.embeddings.state_dict(
        )['word_embeddings.weight']
        self.recnn = torch.nn.Sequential(nn.Linear(768, 768),
                                         nn.Linear(768, 768),
                                         nn.Linear(768, 768))

    def mean_pooling(self, model_output):
        return model_output.mean(axis=1)

    def forward(self, word_ids, token_type_ids, input_ids, attention_mask):
        with torch.no_grad():
            explanation = self.bert(token_type_ids=token_type_ids,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask)[0]
        recnn_output = self.recnn(explanation)
        pred_embed = self.mean_pooling(recnn_output)
        loss = F.mse_loss(pred_embed, self.embedding_weight[word_ids].to(pred_embed.device))
        return {'loss': loss, 'pred_embed': pred_embed}


model = DictNet()

# In[280]:

# - [ ] test multi inputs(outputs is wrong, should be batched.)
# ``` python
# mul_inputs = tokenizer(["I am a bird."], ["He is a chicken."], return_tensors='pt')
# model(mul_inputs).shape
# ```

# In[348]:

# In[310]:

embeddings_weights.shape

# In[312]:

tokenizer.convert_tokens_to_ids('owl')

# In[347]:

# batched inputs
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer,
                                        padding=True,
                                        return_tensors='pt')


def tokenize_function(example):
    res = tokenizer(example['explanation'], truncation=True, padding=True)
    res['word_ids'] = tokenizer.convert_tokens_to_ids(example['word'])
    return res


tokenized_dataset = wordnet_dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(['explanation', 'word'])

data_collator([tokenized_dataset['train'][i] for i in range(8)])

# In[363]:

from transformers import Trainer, TrainingArguments

args = TrainingArguments(output_dir="recnn_test",
                         per_device_train_batch_size=8,
                         num_train_epochs=1,
                         lr_scheduler_type="cosine",
                         learning_rate=5e-4)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    #     data_collator=data_collator
)
# trainer.train()

# In[361]:

from torch.utils.data import DataLoader

tokenized_dataset.set_format("torch")
train_dataloader = DataLoader(tokenized_dataset['train'],
                              shuffle=False,
                              batch_size=8)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

from transformers import get_scheduler

num_epochs = 10
num_train_step = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler('linear',
                             optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=num_train_step)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

model.to(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_train_step))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.set_description(f"loss: {loss:.5f}")
        progress_bar.update(1)
