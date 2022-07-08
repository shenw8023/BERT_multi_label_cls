from transformers import BertModel, BertConfig
import torch
from torch import nn




class BertMultiLabelClassification(nn.Module):
    def __init__(self, config):
        super(BertMultiLabelClassification, self).__init__()
        bert_config = BertConfig.from_pretrained(config.pretrain_path)
        self.bert = BertModel(bert_config)
        self.dropout = nn.Dropout(config.bert_dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size, config.num_labels) 
        

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
