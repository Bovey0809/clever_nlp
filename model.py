import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import BertModel, BertTokenizer


class CleverNLP(nn.Module):
    """Some Information about CleverNLP"""
    def __init__(self, model='bert-base-uncased', device='cuda'):
        super(CleverNLP, self).__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(model)

        self.recnn = torch.nn.Sequential(nn.Linear(768, 768),
                                         nn.Linear(768, 768),
                                         nn.Linear(768, 768))

    def forward(self, sentence, index, explanation):
        sentence = self.tokenizer(sentence,
                                  return_tensors='pt').to(self.device)
        explanation = self.tokenizer(explanation,
                                     return_tensors='pt').to(self.device)

        sentence = self.bert(**sentence)
        embeddings = sentence[0][0]
        original_embed = embeddings[index]

        explanation = self.bert(**explanation)[0][0]
        pred_embed = self.recnn(explanation).sum(0)

        assert original_embed.shape == pred_embed.shape
        loss = F.mse_loss(original_embed, pred_embed)
        return loss