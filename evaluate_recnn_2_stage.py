import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorWithPadding

from dataset import build_dataset, filter_words_by_norm
from model import DictNet
from datetime import datetime


def norm_recall(example_embeddings, original_embeddings, K=10):
    res = []
    distances = []
    for embed in example_embeddings:
        distance = ((original_embeddings - embed)**2).sum(1)
        distances.append(distance)
        indexes = distance.argsort()[:K]
        # print("norm distance: ", distance[indexes])
        res.append(indexes)
    indexes = torch.stack(res)
    distances = torch.stack(distances)
    return indexes, distances


def cosine_recall(example_embedding, original_embeddings, K=10):
    # calculate cosine distance
    cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    indexes = []
    distances = []
    for embed in example_embedding:
        cosine_similarity_of_example = cosine_sim(original_embeddings, embed)
        index = cosine_similarity_of_example.argsort(descending=True)[1:K + 1]
        distance = cosine_similarity_of_example[index]
        # print("cosine distance: ", cosine_similarity_of_example[index])
        indexes.append(index)
        distances.append(distance)
    return torch.stack(indexes), torch.stack(distances)


def original_recall(word, query_embeddings, tokenizer):
    word_id = tokenizer.convert_tokens_to_ids(word)
    word_embeddings = query_embeddings[word_id]
    indexes, distances = cosine_recall(word_embeddings, query_embeddings)
    res = []
    for index in indexes:
        res.append(tokenizer.convert_ids_to_tokens(index))
    return res, distances


def evaluate(words, pred_embeddings, query_embeddings, tokenizer):
    norm_indexes, norm_distances = norm_recall(pred_embeddings,
                                               query_embeddings)
    cos_indexes, cosine_distances = cosine_recall(pred_embeddings,
                                                  query_embeddings)
    original_res, original_distances = original_recall(words, query_embeddings,
                                                       tokenizer)
    result = {}
    for word, norm_id, cos_id, ori_related_words, norm_distance, cosine_distance, original_distance in zip(
            words, norm_indexes, cos_indexes, original_res, norm_distances,
            cosine_distances, original_distances):
        norm_res = tokenizer.convert_ids_to_tokens(norm_id)
        cos_res = tokenizer.convert_ids_to_tokens(cos_id)
        result[word] = {
            'norm_recall': norm_res,
            'norm_distance': norm_distance.cpu().numpy(),
            'cos_recall': cos_res,
            'cos_distance': cosine_distance.cpu().numpy(),
            'original_recall': ori_related_words,
            'original_distance': original_distance.cpu().numpy()
        }

    return result


if __name__ == "__main__":
    # %%
    dict_weight = '/diskb/houbowei/ray_results/train_2022-01-05_20-09-55/train_64e33_00107_107_batch_size=8,epochs=100,lr=0.001_2022-01-05_20-16-11/checkpoint_000098/checkpoint'
    state_dict = torch.load(dict_weight)

    model = DictNet()
    model.load_state_dict(state_dict=state_dict[0])

    # embeddings_weights = "./recnn_stage2_epoch_1_12012022-14-06-24.pt"
    # embeddings_weights = torch.load(embeddings_weights).state_dict()['weight']

    embeddings_weights = model.embedding_weight

    eval_dataset, tokenizer = build_dataset((0, 2))

    # %%
    collate_function = DataCollatorWithPadding(tokenizer)
    eval_dataloader = DataLoader(eval_dataset['train'],
                                 collate_fn=collate_function,
                                 batch_size=8)
    wordnet = pd.read_csv("./wordnet_bert_common_words.csv")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    # %%
    all_words_return = []
    for inputs in tqdm(eval_dataloader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        preds = model(**inputs).detach()
        words = inputs['word_ids']
        words = tokenizer.convert_ids_to_tokens(words)
        results = evaluate(words, preds, embeddings_weights.to(device),
                           tokenizer)
        for k, v in results.items():
            definition = str(
                wordnet[wordnet.words == k].definition.values).strip('[]\'"')

            # print(f"{k}: {definition}")
            v['definition'] = definition
            v['word'] = k
            all_words_return.append(v)
            # for recall, related_words in v.items():
            #     print(f"{recall} : {related_words}")
            # print('\n')
    # %%
    output_csv = pd.DataFrame(all_words_return)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    filename = f'wordnet_results-{dt_string}.csv'
    output_csv.to_csv(filename)
    print(f"results save at {filename}")