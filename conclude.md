
- 错误做法
模型继承自nn.Module
在模型中 self.bert=BertModel.from_pretrained(pretrain_path)
因为虽然在训练过程没有问题，但是在预测阶段，如果复用model定义这份代码的话，在初始化模型实例的时候，运行到上面这句，就要求pretrain_path必须包含pytorch_model.bin这个bert原始模型的ckpt；而我们实际部署的时候只会提供我们自己训练完的save_model.ckpt；所以只能重写模型定义。


- 正确做法1：
继承自transformers.BertPreTrainedModel
模型初始化参数是BertConfig实例，self.model=BertModel(config)
这样模型就继承了transformers模型的一些方法，
这样在训练阶段，使用其from_pretrained('bert-base-chinese')加载预训练模型
在部署预测阶段，使用其from_pretrained('bert-base-chinese', state_dict=model_state_dict))， 通过state_dict参数指定我们自己保存的模型
    或者使用save_pretrained(save_path)方法保存，再使用from_pretrained(save_path)加载整个模型参数


- 正确做法2：（更妥）
模型继承自nn.Module
在模型中 self.bert=BertModel(config=BertConfig(bert_config_path))
训练阶段：单独对模型bert层加载预训练参数：
    model.bert.load_state_dict(torch.load("pytorch_model.bin"))  或者
    model.bert = model.bert.from_pretrained(pretrian_path)
部署预测阶段：对整个模型加载保存的参数：model.load_state_dict(torch.load(save_path))


- weight_decay
- lr_decay
- save ckpt, continue train
- logger
- warmup


_, pooler_output = self.model(**inputs, return_dict=False)
- BCEWithLogitsLoss(pred, target) 二者都必须为FloatTensor
- 必须将模型的参数加载完了，再to(device)，否则的话，加载的参数是没法同步到device上