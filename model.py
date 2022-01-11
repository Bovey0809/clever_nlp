import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel


class RecNN(nn.Module):
    def __init__(self, in_features=768) -> None:
        super(RecNN, self).__init__()
        self.l1 = nn.Linear(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.act1 = nn.GELU()
        self.l2 = nn.Linear(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.act2 = nn.GELU()

        self.pool = nn.AvgPool1d(kernel_size=1, padding=0, stride=1)

    def forward(self, embeddings):
        embeddings = self.l1(embeddings)
        embeddings = embeddings.transpose(2, 1)
        embeddings = self.bn1(embeddings)
        embeddings = embeddings.transpose(2, 1)
        embeddings = self.act1(embeddings)

        embeddings = self.l2(embeddings)
        embeddings = embeddings.transpose(2, 1)
        embeddings = self.bn2(embeddings)

        embeddings = embeddings.transpose(2, 1)
        # embeddings = self.act2(embeddings)

        # embeddings = embeddings.transpose(2, 1)
        # embeddings = self.pool(embeddings)

        return embeddings.mean(axis=1)


class DictNet(nn.Module):
    """Some Information about CleverNLP"""
    def __init__(self, model='bert-base-uncased', device='cuda'):
        super(DictNet, self).__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(model)
        self.embedding_weight = self.bert.embeddings.state_dict(
        )['word_embeddings.weight']
        self.recnn = RecNN()
        for param in self.bert.parameters():
            param.requires_grad = False

    def mean_pooling(self, model_output):
        return model_output.mean(axis=1)

    def forward_train(self, word_ids, token_type_ids, input_ids,
                      attention_mask):
        # STEP1: freeze bert

        explanation = self.bert(token_type_ids=token_type_ids,
                                input_ids=input_ids,
                                attention_mask=attention_mask)[0]
        pred_embed = self.recnn(explanation)
        loss = F.mse_loss(
            pred_embed, self.embedding_weight[word_ids].to(pred_embed.device))

        return {'loss': loss, 'pred_embed': pred_embed}

    def forward_test(self, word_ids, token_type_ids, input_ids,
                     attention_mask):
        explanation = self.bert(token_type_ids=token_type_ids,
                                input_ids=input_ids,
                                attention_mask=attention_mask)[0]
        pred_embed = self.recnn(explanation)
        return pred_embed

    def forward(self, word_ids, token_type_ids, input_ids, attention_mask):
        if self.training:
            return self.forward_train(word_ids, token_type_ids, input_ids,
                                      attention_mask)
        else:
            return self.forward_test(word_ids, token_type_ids, input_ids,
                                     attention_mask)


class BertRecNN(nn.Module):
    def __init__(self, model='bert-base-uncased', device='cuda') -> None:
        super(BertRecNN, self).__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(model)
        self.freeze_weights()

    def forward(self, inputs):
        word_ids = inputs.pop('word_ids')
        return self.bert(**inputs)

    def freeze_weights(self):
        """Only Embeddings layers should be trainable."""
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.embeddings.parameters():
            param.requires_grad = True

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings = value