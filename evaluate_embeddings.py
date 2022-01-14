# %%
import torch
from model import DictNet
from transformers import AutoModel, AutoTokenizer
from evaluation import cosine_recall, norm_recall
from dataset import build_embeddings_dataset, filter_words_by_norm

# low frequency dict
filtered_words = filter_words_by_norm((1.35, 2))

# original embeddings
checkpoint = "bert-base-uncased"
model = AutoModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
original_word_embeddings = model.embeddings.word_embeddings.state_dict(
)['weight']

# new word embeddings
new_embeddings = "./recnn_stage2_epoch_0_12012022-14-06-03.pt"
new_embeddings = torch.load(new_embeddings).state_dict()['weight']

# Select low, middle, high words by norm
words = []
for word in filtered_words:
    if isinstance(word, str):
        words.append(word)

selected_words_ids = tokenizer.convert_tokens_to_ids(words)

# %%
# get the norms of the words
original_words_norms = original_word_embeddings[selected_words_ids].norm(dim=1)
new_words_norms = new_embeddings[selected_words_ids].norm(dim=1)

# %%
original_sort_ids = sorted(selected_words_ids,
                           key=lambda x: original_word_embeddings[x].norm())
# %%
import random

high_freq_words = random.choices(original_sort_ids[:2231], k=10)
middle_freq = random.choices(original_sort_ids[2231:4462], k=10)
low_freq = random.choices(original_sort_ids[4462:], k=10)
# %%
# corresponding words
ori_high_words = tokenizer.convert_ids_to_tokens(high_freq_words)
# %%
ori_middle_words = tokenizer.convert_ids_to_tokens(middle_freq)
# %%
ori_low_words = tokenizer.convert_ids_to_tokens(low_freq)
# %%
res_words_ids, _ = cosine_recall(
    original_word_embeddings[low_freq[0]].unsqueeze(0),
    original_word_embeddings)
# %%
words = [
    'amir', 'convergence', 'nutritional', 'greco', 'melee', 'psalm', 'ee',
    'annals', 'cambrian', 'triad'
    'migrant', 'detrimental', 'experimentation', 'suv', 'abolitionist',
    'scorpion', 'sincerely', 'engraved', 'descend', 'labyrinth'
    'concealed', 'flemish', 'gaping', 'ornamental', 'cpu', 'derivative',
    'amulet', 'alt', 'pickup', 'imposing'
]


# %%
def build_dict(words, full_embeddings):
    word_ids = tokenizer.convert_tokens_to_ids(words)
    word_embeddings = full_embeddings[word_ids]
    word_norms = word_embeddings.norm(dim=1)
    cosine_recall_ids, cosine_distance = cosine_recall(word_embeddings,
                                                     full_embeddings)
    recall_words = []
    for recall_ids in cosine_recall_ids:
        recall_words.append(tokenizer.convert_ids_to_tokens(recall_ids))

    res_dict = dict(words=list(words),
                    recalls=recall_words,
                    norms=word_norms.cpu().numpy())
    return res_dict


# %%
word_ids = tokenizer.convert_tokens_to_ids(words)
# %%
word_norms = original_word_embeddings[word_ids].norm(dim=1)


# %%
def filter_words_by_norm(norm_range=(0, 1.2)):
    data = pd.read_csv("wordnet_bert_common_words.csv")
    words = data[data.norms.apply(
        lambda x: norm_range[0] < x < norm_range[1])].words.values
    words = set(words.tolist())
    print(f">>>> There are {len(words)} words between norm range {norm_range}")
    return words


# %%

# add all the words in definiton to definition_set
import pandas as pd

data = pd.read_csv("./wordnet_bert_common_words.csv")
# %%
definiton_series = data.definition.apply(lambda x: x.split())
# %%
definition_set = set()
for defition in definiton_series:
    definition_set = definition_set | set(defition)
# DONE Get all the words from definitons.
# %%
# get words by the range norm (1.35, 2)
selected_words = filter_words_by_norm((1.35, 2))
# %%
selected_words_set = set()
for word in selected_words:
    if isinstance(word, str):
        selected_words_set.add(word)
# %%
print(len(definition_set & selected_words_set), " words in the dict")
# %%
filtered_words_set = definition_set & selected_words_set
# %%
original_words_recall_dict = build_dict(filtered_words_set,
                                        original_word_embeddings)
new_words_recall_dict = build_dict(filtered_words_set, new_embeddings)


# %%
def calculate_norms(words, embeddings) -> dict:
    ids = tokenizer.convert_tokens_to_ids(words)
    norms = embeddings[ids].norm(dim=1).cpu().numpy()
    return dict(zip(words, norms))


# %%
ori_norms_dict = calculate_norms(filtered_words_set, original_word_embeddings)
# %%
new_norms_dict = calculate_norms(filtered_words_set, new_embeddings)
# %%
orgi_df = pd.DataFrame(original_words_recall_dict)
# %%
new_df = pd.DataFrame(new_words_recall_dict)
# %%
result_dict = pd.merge(orgi_df,
                       new_df,
                       on='words',
                       suffixes=("_original", "_new"))
# %%
# result_dict.to_csv("orginal_recnn_embeddings_compare_epoch1.csv")
result_dict[result_dict.words == "gland"]
# %%

# %%
