from transformers import BertTokenizer, BertModel

import argparse
import warnings

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Replace me by any text you'd like.")
    args = parser.parse_args()
    text = args.text

    print(f"Input Text: {text} \n")

    # STEP1 : Word --> tokens
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(text, return_tensors='pt')

    # Check encoding
    print("Parse Text to tokens")
    tokens = []
    for i in encoded_input['input_ids']:
        tokens.append(tokenizer.convert_ids_to_tokens(i))
    print(f'Tokens: {tokens} \n')
    model = BertModel.from_pretrained("bert-base-uncased")

    output = model(**encoded_input)

    # Tokens -> Embeddings

    print(output)


if __name__ == "__main__":
    main()