# %%
from transformers.data import DataCollatorWithPadding
from torch.utils.data.dataloader import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer
from dataloader import build_dataset, word_dict

from model import DictNet
import torch
import pandas as pd


def norm_recall(example_embeddings, original_embeddings, K=10):
    res = []
    for embed in example_embeddings:
        distance = ((original_embeddings - embed)**2).sum(1)
        indexes = distance.argsort()[:K]
        res.append(indexes)
    indexes = torch.stack(res)
    return indexes


def cosine_recall(example_embedding, original_embeddings, K=10):
    # calculate cosine distance
    cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    indexes = []
    for embed in example_embedding:
        cosine_similarity_of_example = cosine_sim(original_embeddings, embed)

        index = cosine_similarity_of_example.argsort(descending=True)[:K]
        indexes.append(index)
    return torch.stack(indexes)


def original_recall(word, query_embeddings, tokenizer):
    word_id = tokenizer.convert_tokens_to_ids(word)
    word_embeddings = query_embeddings[word_id]
    indexes = cosine_recall(word_embeddings, query_embeddings)
    res = []
    for index in indexes:
        res.append(tokenizer.convert_ids_to_tokens(index))
    return res


def evaluate(words, pred_embeddings, query_embeddings, tokenizer):
    norm_indexes = norm_recall(pred_embeddings, query_embeddings)
    cos_indexes = cosine_recall(pred_embeddings, query_embeddings)
    original_res = original_recall(words, query_embeddings, tokenizer)
    result = {}
    for word, norm_id, cos_id, ori_related_words in zip(
            words, norm_indexes, cos_indexes, original_res):
        norm_res = tokenizer.convert_ids_to_tokens(norm_id),
        cos_res = tokenizer.convert_ids_to_tokens(cos_id)
        result[word] = {
            'norm_recall': norm_res,
            'cos_recall': cos_res,
            'original_recall': ori_related_words
        }

    return result


# %%
weight_path = '/diskb/houbowei/ray_results/train_2022-01-05_20-09-55/train_64e33_00107_107_batch_size=8,epochs=100,lr=0.001_2022-01-05_20-16-11/checkpoint_000098/checkpoint'
state_dict = torch.load(weight_path)

model = DictNet()
model.load_state_dict(state_dict=state_dict[0])

eval_dataset, tokenizer = build_dataset((1.3, 2))

# %%
collate_function = DataCollatorWithPadding(tokenizer)
eval_dataloader = DataLoader(eval_dataset['train'],
                             collate_fn=collate_function,
                             batch_size=16)
wordnet = pd.read_csv("./wordnet_bert_common_words.csv")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
# %%
for inputs in eval_dataloader:
    inputs = {k: v.to(device) for k, v in inputs.items()}
    preds = model(**inputs)
    words = inputs['word_ids']
    words = tokenizer.convert_ids_to_tokens(words)
    results = evaluate(words, preds, model.embedding_weight.to(device),
                       tokenizer)
    for k, v in results.items():
        definition = str(
            wordnet[wordnet.words == k].definition.values).strip('[]\'"')

        print(f"{k}: {definition}")
        for recall, related_words in v.items():
            print(f"{recall} : {related_words}")
        print('\n')
# %%
