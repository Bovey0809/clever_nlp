# %%
from transformers import AutoTokenizer

corpus = [
    "This is the Hugging Face course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%
from collections import defaultdict
# %%
word_freqs = defaultdict(int)
# %%
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1
print(word_freqs)
# %%
alphabet = []
for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()
# %%
vocab = ["<|endoftext|>"] + alphabet.copy()
# %%
splits = {word: [c for c in word] for word in word_freqs.keys()}


# %%
def compute_pair_freqs(splits):
    """
    compute_pair_freqs calculate freq of pairs in each words.

    pairs of two characters.

    Args:
        splits (dict): word:characters.

    Returns:
        dict: pair:freq
    """
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


# %%
pair_freqs = compute_pair_freqs(splits)
# %%
for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}:{pair_freqs[key]}")
    if i >= 5:
        break
# %%
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq
print(best_pair, max_freq)
# %%
merges = {best_pair:''.join(best_pair)}
# %%
vocab.append("".join(best_pair))
# %%
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i+1] == b:
                split = split[:i] + [a+b] + split[i+2:]
            else:
                i += 1
        splits[word] = split
    return splits
# %%
splits = merge_pair(best_pair[0], best_pair[1], splits)
# %%
vocab_size = 50
while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, frreq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0]+best_pair[1])
# %%
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])
# %%
tokenize("This is not a token.")
# %%
from datasets import load_dataset
# %%
dataset = load_dataset("wikitext", name='wikitext-2-raw-v1', split='train')
# %%
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i:i+1000]['text']
# %%
with open("wikitext-2.txt", "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        f.write(dataset[i]['text']+'\n')
# %%
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
# %%
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
# %%
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
# %%
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
# %%
print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
# %%

tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# %%
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# %%
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
# %%
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
# %%
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
# %%
tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
# %%
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
# %%
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)
# %%
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)
# %%
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
# %%
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)
# %%
