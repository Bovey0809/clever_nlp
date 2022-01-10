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


def nltk_dataset(
        filepath='/diskb/houbowei/clever_nlp/wordnet_bert_common_words.csv'):
    '''Read csv file and return nltk dataset.
    '''
    dataset = load_dataset('csv', data_files=filepath)
    return dataset


def build_dataset(range=(0, 1.233141)) -> tuple:
    """
    build_dataset build transformers tokenized dataset.
            norms
    count	14510.00000
    mean	1.339519	
    std	0.151899	
    min	0.866191	
    25%	1.233141	
    50%	1.355623	
    75%	1.452249	
    max	1.823229

    Args:
        range: norm range for filtering.

    Raises:
        NotImplemented: only nltk and core dataset are supported.

    Returns:
        dataset: dataset from transformers.
    """
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = nltk_dataset()
    dataset = dataset.filter(lambda x: range[0] < x['norms'] < range[1])
    tokenized_dataset = dataset.map(lambda x: tokenizer(x['definition']),
                                    batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(
        ['definition', 'words', 'embeddings_ids', 'norms'])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset, tokenizer
