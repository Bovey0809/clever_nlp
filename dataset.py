import pandas as pd

from datasets import load_dataset

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
        tokenizer: tokenizer for the dataset.
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


def build_embeddings_dataset(norm_range=(0, 1.233141)):
    """
    Build dataset for embeddings.

    The dataset containing definitions from wordnet.
    And each definition must include the words which the norm within the range.

    Args:
        range (tuple, optional): norm range. Defaults to (0, 1.233141).
    
    Returns:
        dataset: columns for bert including target_word_position.
        tokenizer: tokenizer for the dataset.
    """
    assert norm_range[0] <= norm_range[1], "Range should be (min, max) format."
    checkpoint = "bert-base-uncased"
    print(">>>>>>> Downloading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = nltk_dataset()
    words_in_range = filter_words_by_norm(norm_range)

    def _filter_definition(example):
        definition = example['definition']
        for i in definition.split():
            if i in words_in_range:
                return True
        return False

    def _target_word_position(example):
        """
        _target_word_position Generate tokens index for dataset.

        tokens[target_index] is the definition.

        Args:
            example (dict): dataset inputs dict.

        Returns:
            dict: bert inputs & target_word_position.
        """
        definition = example['definition']
        words_position = []

        res = tokenizer.encode_plus(definition)
        tokens = tokenizer.convert_ids_to_tokens(res['input_ids'])
        for index, word in enumerate(tokens):
            if word in words_in_range:
                words_position.append(index)
        res['target_word_position'] = words_position
        return res

    print(">>> filtering dataset by norm range.")
    dataset = dataset.filter(_filter_definition)
    # TODO the _target_word_postion is not ready for BATCH!
    tokenized_dataset = dataset.map(_target_word_position)
    tokenized_dataset = tokenized_dataset.remove_columns(
        ['definition', 'words', 'embeddings_ids', 'norms', 'word_ids'])
    # tokenized_dataset.set_format(
    #     "torch",
    #     columns=['word_ids', 'input_ids', 'token_type_ids', 'attention_mask'])
    return tokenized_dataset, tokenizer