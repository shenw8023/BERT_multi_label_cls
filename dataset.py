import json
from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_label_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        labels = f.read().strip().split("\n")
    label2id, id2label = {}, {}
    for i, l in enumerate(labels):
        label2id[l] = i
        id2label[i] = l
    return label2id, id2label, labels




def read_json_data(file_path):
    """
    多种类型文件读取方式
    return: 
        texts:list  每项为一个句子
        labels:list 每项为所有相关标签
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = f.read().strip().split("\n")
    texts, labels = [], []
    for line in raw_data:
        line_ = json.loads(line)
        
        if len(line_['event_list']) != 0:
            label = []
            for tmp in line_['event_list']:
                label.append(tmp['event_type'])
            texts.append(line_['text'])
            labels.append(label)

    return texts, labels





def load_dataset(config, read_func, mode='train'):
    if mode=='train':
        file_path = config.train_path 
    elif mode=='dev':
        file_path = config.dev_path
    elif mode=='test':
        file_path = config.test_path

    texts, labels = read_func(file_path)
    assert len(texts) == len(labels), "读取数据条数和对应标签条数不一致"
    
    # tokenize
    if config.pretrain_path:
        tok = BertTokenizer.from_pretrained(config.pretrain_path) 
    else:
        tok = BertTokenizer.from_pretrained('bert-base-chinese')
    tensor_inputs = tok(texts, padding='max_length', truncation=True, max_length=config.max_seq_len, return_tensors='pt') 
    
    # one_hot_labels
    _, _, classes = load_label_dict(config.label_path)
    mlb = MultiLabelBinarizer(classes=classes)
    one_hot_labels = mlb.fit_transform(labels)
    one_hot_labels = torch.FloatTensor(one_hot_labels)

    return tensor_inputs, one_hot_labels
    


# class MyDataset(Dataset):
#     def __init__(self, texts, labels, config):
#         self.texts = texts
#         self.labels = labels
#         self.config = config
#     def __getitem__(self, indx):
#         return self.texts[indx], self.labels[indx]
    
#     def __len__(self):
#         return len(self.texts)



def gen_dataloader(config, read_func, mode='train'):
    shuffle_ = True if mode=='train' else False
    inputs, labels = load_dataset(config, read_func, mode=mode)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle_) 
    return dataloader


