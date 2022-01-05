from transformers import AutoModel, AutoTokenizer

bert_model = AutoModel.from_pretrained('bert-base-uncased')

# %%
original_embeddings = bert_model.embeddings.word_embeddings.state_dict(
)['weight']

# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# %%
original_norms = original_embeddings.norm(dim=1)

K = 10

from dataloader import word_dict

# %%
wordnet = word_dict()

# %%
common_words = list(set(wordnet.keys()) & set(tokenizer.vocab.keys()))

# %% [markdown]
# 计算common words 中的每个词汇的 embeddings, 找出低频词

# %%
common_words_norms = {
    i: original_norms[tokenizer.convert_tokens_to_ids(i)]
    for i in common_words
}

# %%
# 最长的10个norms的单词
K = 20
common_words_norms_sorted = dict(
    sorted(common_words_norms.items(), key=lambda item: item[1], reverse=True))

example = "wrestle"
# gallon's embeddings
example_embedding = original_embeddings[tokenizer.convert_tokens_to_ids(
    example)]

# calculate cosine distance
import torch

cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

cosine_similarity_of_example = cosine_sim(original_embeddings,
                                          example_embedding)

tokenizer.convert_ids_to_tokens(
    cosine_similarity_of_example.argsort(descending=True)[:K])

# %% [markdown]
# 上面的词汇大部分的都是空的, 再对高频词做同样的处理, 验证

# %%
example = "woman"
# gallon's embeddings
example_embedding = original_embeddings[tokenizer.convert_tokens_to_ids(
    example)]

# calculate cosine distance
import torch

cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

cosine_similarity_of_example = cosine_sim(original_embeddings,
                                          example_embedding)

tokenizer.convert_ids_to_tokens(
    cosine_similarity_of_example.argsort(descending=True)[:K])


# %%
def show_original_related_words(example):
    example_embedding = original_embeddings[tokenizer.convert_tokens_to_ids(
        example)]
    # calculate cosine distance
    cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    cosine_similarity_of_example = cosine_sim(original_embeddings,
                                              example_embedding)

    return tokenizer.convert_ids_to_tokens(
        cosine_similarity_of_example.argsort(descending=True)[:K])


show_original_related_words('woman')

# %% [markdown]
# 对于低频词, embeddings的语义相关单词较少, 对于高频词, 语义更加丰富.
#
# 训练recnn网络, 通过网络调整低频词embeddings的效果


# %%
def calculate_topK_distance_by_mse(embeddings, pred_embeddings, K=10):
    return ((embeddings - pred_embeddings)**2).sum(axis=0)


# %%
from model import DictNet

recnn = torch.load("./recnn-last.pt")


def get_related_words_from_recnn(example):
    try:
        input_sentence = wordnet[example]
    except KeyError as e:
        return "words should be in the wordnet dictionary."
    print("Input Definition:", input_sentence)
    res = tokenizer(input_sentence, return_tensors='pt').to('cuda')

    res['word_ids'] = torch.tensor(
        tokenizer.convert_tokens_to_ids(example)).to('cuda')
    recnn.eval()
    output = recnn(**res)

    new_gallon_embed = output['pred_embed'].to('cpu')
    print(f"{example}: {new_gallon_embed.norm()}")
    cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    cosine_similarity_of_example = cosine_sim(original_embeddings,
                                              new_gallon_embed)
    print(cosine_similarity_of_example.argsort(descending=True)[:K])
    print("cosine:", tokenizer.convert_ids_to_tokens(
        cosine_similarity_of_example.argsort(descending=True)[:K]))
    return tokenizer.convert_ids_to_tokens(
        calculate_topK_distance_by_mse(original_embeddings,
                                       new_gallon_embed).argsort()[:K])

# original nrom
original_norms[tokenizer.convert_tokens_to_ids(example)]

# %% [markdown]
# 训练之后的模长变短了, 但是语义相近的词并不对

# %%
example = 'bird'
print(get_related_words_from_recnn(example))
print("original norm",
      original_norms[tokenizer.convert_tokens_to_ids(example)])
print("\n")
print(show_original_related_words(example))
