from pandas.core.frame import DataFrame
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
import torch
import argparse
import warnings
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random

warnings.filterwarnings('ignore')


def draw(embeddings, ids=None, figsize=(30, 10)):
    df = draw_tsne(embeddings)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax1.scatter(df.x, df.y, alpha=.1)
    # ax1.set_yticklabels(ids)

    norms = draw_norm(embeddings)
    ax2.bar(x=range(len(norms)), height=norms)
    ax2.set_xticks(range(len(norms)))
    ax2.set_xticklabels(ids)
    # ax2.set_xlabel(ids)
    plt.show()


def draw_tsne(embeddings):
    embeddings = embeddings.detach().cpu().numpy()
    tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
    embs = tsne.fit_transform(embeddings)
    df = DataFrame()
    df['x'] = embs[:, 0]
    df['y'] = embs[:, 1]
    return df


def draw_norm(embeddings):
    norms = torch.norm(embeddings, dim=1).detach().cpu().numpy()
    return norms


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0]  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, -1)
    sum_mask = torch.clamp(input_mask_expanded.sum(-1), min=1e-9)
    return sum_embeddings / sum_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Replace me by any text you'd like.")
    args = parser.parse_args()
    text = args.text
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Input Text: {text} \n")

    # STEP1 : Word --> tokens
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(text, return_tensors='pt').to(device)

    # Check encoding
    print("Parse Text to tokens")
    tokens = []
    for i in encoded_input['input_ids']:
        tokens.extend(tokenizer.convert_ids_to_tokens(i))
    print(f'Tokens: {tokens} \n')
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    output = model(**encoded_input)

    # Tokens -> Embeddings
    embeddings = output[0][0]
    assert embeddings.shape[0] == len(
        tokens), "Token size should be the same as embeddings."

    # Draw tsne for testing.
    draw(embeddings, ids=tokens)
    return embeddings


if __name__ == "__main__":
    main()