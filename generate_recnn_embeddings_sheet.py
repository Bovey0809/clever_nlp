# %%
from datetime import datetime
import pickle
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from transformers.data import DataCollatorWithPadding

from dataset import build_dataset
from model import DictNet


def main(weight_path):
    state_dict = torch.load(weight_path)

    model = DictNet()
    model.load_state_dict(state_dict=state_dict[0])

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

    words_embeddings = {}

    for inputs in tqdm(eval_dataloader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        pred_embeddingss = model(**inputs).detach()
        words = inputs['word_ids']
        words = tokenizer.convert_ids_to_tokens(words)
        words_embeddings.update(dict(zip(words, pred_embeddingss.detach().cpu().numpy())))

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y-%H-%M-%S")
    filename = f"words_preembeddings_dict-{dt_string}.p" 
    with open(filename, "wb") as f:
        pickle.dump(words_embeddings, f)
    print(f"The word embeddings is saved as {filename}")


if __name__ == "__main__":
    weight_path = '/diskb/houbowei/ray_results/train_2022-01-05_20-09-55/train_64e33_00107_107_batch_size=8,epochs=100,lr=0.001_2022-01-05_20-16-11/checkpoint_000098/checkpoint'
    main(weight_path)