import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, AutoModel


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
        # Normalization for embeddings
        normalized_embedding_weights = dict(
            weight=(self.embedding_weight - self.embedding_weight.mean()) /
            self.embedding_weight.std())
        self.bert.embeddings.word_embeddings.load_state_dict(
            normalized_embedding_weights)
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

    def forward(self, bert_inputs, target_position):
        """
        forward Stage 2 forward function.

        1. Freeze all layers in Bert except embeddings.
        2. Extract last hidden layer.
        3. Select specific position words according to norm range.

        Args:
            bert_inputs (dict): original bert inputs with word ids.
            target_position (list[int]): word position in the inputs'id.

        Returns:
            bert_output: Last layer's output vector.
            word_ids: Vector ids.
        """
        input_ids = bert_inputs['input_ids']
        last_hidden_state = self.bert(**bert_inputs)['last_hidden_state']
        bert_output_vectors = []
        # Word ids is the id in vocabulary.
        word_ids = []
        for batch_id, output in enumerate(last_hidden_state):
            position = target_position[batch_id]
            bert_output_vectors.append(output[position])
            word_id = input_ids[batch_id][position]
            word_ids.append(word_id)
        bert_output_vectors = torch.cat(bert_output_vectors)
        return bert_output_vectors, word_ids

    def freeze_weights(self):
        """Only Embeddings layers should be trainable."""
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.embeddings.word_embeddings.parameters():
            param.requires_grad = True

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings = value


class ChineseDictNet(DictNet):

    def __init__(self,
                 model="hfl/chinese-bert-wwm-ext",
                 device='cuda') -> None:
        super(ChineseDictNet, self).__init__(model=model, device=device)

    def forward_train(self, word_ids, token_type_ids, input_ids,
                      attention_mask):
        explanation = self.bert(token_type_ids=token_type_ids,
                                input_ids=input_ids,
                                attention_mask=attention_mask)[0]

        pred_embed = self.recnn(explanation)

        # Chinese seperate sentence into words
        chinese_word_embeddings = []
        for word_id in word_ids:
            chinese_word_embedding = self.embedding_weight[word_id].mean(
                axis=0)
            chinese_word_embeddings.append(chinese_word_embedding)
        chinese_word_embeddings = torch.stack(chinese_word_embeddings).to(pred_embed.device)
        loss = F.mse_loss(pred_embed, chinese_word_embeddings)
        return {'loss': loss, 'pred_embed': pred_embed}
