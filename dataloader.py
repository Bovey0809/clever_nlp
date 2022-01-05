from datasets.arrow_dataset import Batch
import pandas as pd

from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer


def word_dict(data_dir="/diskb/houbowei/clever_nlp/core-wordnet.txt") -> dict:
    data_files = data_dir

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


def nltk_dataset(filepath='./wordnet_bert_common_words.csv'):
    '''Read csv file and return nltk dataset.
    '''
    dataset = load_dataset('csv', data_files=filepath)
    return dataset


def build_dataset(name='wordnet_core'):
    """
    build_dataset build transformers tokenized dataset.

    Args:
        name (str, optional): Defaults to 'wordnet_core', optional 'wordnet_nltk".

    Raises:
        NotImplemented: only nltk and core dataset are supported.

    Returns:
        dataset: dataset from transformers.
    """

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if name == 'wordnet_core':
        # Build Dataset

        wordnet = word_dict()
        words_wordnet = list(wordnet.keys())

        words_bert = list(tokenizer.vocab.keys())
        common_words = set(words_bert) & set(words_wordnet)
        common_words_dict = {
            k: v
            for k, v in wordnet.items() if k in common_words
        }
        wordnet_series = pd.DataFrame(
            {
                'word': common_words_dict.keys(),
                'explanation': common_words_dict.values()
            },
            index=None)
        wordnet_series.to_csv("wordnet.csv", index=None)
        wordnet_dataset = load_dataset('csv', data_files="wordnet.csv")

        def tokenize_function(example):
            res = tokenizer(example['explanation'], padding=True)
            res['word_ids'] = tokenizer.convert_tokens_to_ids(example['word'])
            return res

        tokenized_dataset = wordnet_dataset.map(tokenize_function,
                                                batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(
            ['explanation', 'word'])
    elif name == "wordnet_nltk":
        dataset = nltk_dataset()
        tokenized_dataset = dataset.map(lambda x: tokenizer(x['definition']),
                                        batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(
            ['definition', 'words', 'embeddings_ids', 'norms'])
    else:
        raise NotImplemented

    tokenized_dataset.set_format("torch")
    return tokenized_dataset, tokenizer
