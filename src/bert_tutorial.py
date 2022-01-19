# %%
from huggingface_hub import hf_hub_url

data_files = hf_hub_url(
    repo_id="lewtun/github-issues",
    filename="datasets-issues-with-comments.jsonl",
    repo_type="dataset",
)
# %%
from datasets import load_dataset

issues_dataset = load_dataset("json", data_files=data_files, split="train")
issues_dataset
# %%
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0))
issues_dataset
# %%
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
issues_dataset
# %%
issues_dataset.set_format("pandas")
df = issues_dataset[:]
# %%
df["comments"][0].tolist()
# %%
comments_df = df.explode("comments", ignore_index=True)
comments_df.head(4)
# %%
from datasets import Dataset

comments_dataset = Dataset.from_pandas(comments_df)
comments_dataset
# %%
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())})
# %%
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
comments_dataset


# %%
def concatenate_text(examples):
    return {
        "text":
        examples["title"] + " \n " + examples["body"] + " \n " +
        examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_text)
# %%
from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
# %%
import torch

device = torch.device("cuda")
model.to(device)


# %%
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


# %%
def get_embeddings(text_list):
    encoded_input = tokenizer(text_list,
                              padding=True,
                              truncation=True,
                              return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


# %%
embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape
# %%
embeddings_dataset = comments_dataset.map(
    lambda x:
    {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]})

# %%
embeddings_dataset.add_faiss_index(column="embeddings", device=2)
# %%
question = "embeddings?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape

scores, samples = embeddings_dataset.get_nearest_examples("embeddings",
                                                          question_embedding,
                                                          k=10)

import pandas as pd

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()
# %%
