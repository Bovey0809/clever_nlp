# clever_nlp

# Pipeline


```mermaid
graph LR
letters --> Tokens --> embeddings --> TSNE
embeddings --> norm
```

# Model

```mermaid
graph LR
s[\"she feed the owl"\] --> b[bert] --> e[embeddings] 
i[id] --> e --> emb[/embdding/] --> l[loss]
ex[\owl:a bird of prey with large round eyes\] --> b --> r[recnn] --> em[/embedding/] --> l
```
