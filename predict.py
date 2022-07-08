

from config import get_config
from dataset import load_label_dict
from model import BertMultiLabelClassification


import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import time


class ModelPredictor:
    def __init__(self, **kwargs):
        config = get_config()
        for arg, value in kwargs.items():
            config.__setattr__(arg, value)
        self.config = config
        
        device_id = self.config.device_id
        self.device = torch.device('cpu' if device_id==-1 else 'cuda:'+str(device_id))
        _, id2label, labels = load_label_dict(self.config.label_path)
        self.id2label = id2label
        self.labels = labels
        self.config.num_labels = len(labels)
        self.load_model()

        self.tokenizer = BertTokenizer.from_pretrained(self.config.vocab_path)
    
    def load_model(self):
        ckpt = torch.load(self.config.best_model_state_path)
        model = BertMultiLabelClassification(self.config)
        model.load_state_dict(ckpt)
        self.model = model
        self.model.to(self.device)
        self.model.eval()

    def process_data(self, data:list):
        tensor_inputs = self.tokenizer(data, padding='max_length', 
                                        truncation=True, 
                                        max_length=self.config.max_seq_len, 
                                        return_tensors='pt')
        dataset = TensorDataset(tensor_inputs['input_ids'], tensor_inputs['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        return dataloader



    def __call__(self, data:list):
        dataloader = self.process_data(data)
        preds = []
        with torch.no_grad():
            for input_ids, attention_mask in dataloader:
                logits = self.model(input_ids.to(self.device), attention_mask.to(self.device))
                logits = torch.sigmoid(logits).cpu().detach().numpy().tolist()
                outputs = (np.array(logits) > 0.6).astype(int)
                preds.extend(outputs.tolist())

        preds_label = [self.convert_id_label(i) for i in preds]
        return preds_label


    def convert_id_label(self, id_list):
        labels = []
        for idx, pred in enumerate(id_list):
            if pred == 1:
                labels.append(self.id2label[idx])
        return labels



if __name__ == "__main__":
    model = ModelPredictor(device_id=3)  # 所有config中默认的参数都可以重新指定
    data = [
        "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了",
        "前两天，被称为 “ 仅次于苹果的软件服务商 ” 的 Oracle（ 甲骨文 ）公司突然宣布在中国裁员。。",
        "不仅仅是中国IT企业在裁员，为何500强的甲骨文也发生了全球裁员",
        "国家新闻出版署约谈12家存在低俗问题的网络文学企业",
        "近日，有消息称，江苏省消保委结合智能电视开机广告专项消费调查情况，对七家智能电视品牌进行了集体性约谈。",
        
    ]
    t1 = time.time()
    pred = model(data)
    t2 = time.time()
    print(pred)
    print(t2 - t1)