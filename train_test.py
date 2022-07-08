from tabnanny import check
import utils
from config import get_config
from model import BertMultiLabelClassification
from dataset import gen_dataloader, load_label_dict, read_json_data

from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, classification_report
from torch import nn
import torch
import numpy as np
import logging
import os
from tensorboardX import SummaryWriter
import time

logger = logging.getLogger(__name__)



class Trainer:
    def __init__(self, args):
        self.args = args
        if args.device_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:'+str(args.device_id))

        _, _, labels = load_label_dict(args.label_path)
        self.labels = labels
        self.args.num_labels = len(labels)
        
        self.model = BertMultiLabelClassification(self.args)
        self.model.bert = self.model.bert.from_pretrained(self.args.pretrain_path)
        self.model.to(self.device)
        

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.writer = SummaryWriter(log_dir=self.args.log_path + time.strftime('%m-%d_%H.%M', time.localtime())) #TODO



    def train(self, train_loader, dev_loader, checkpoint_path=None):
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)

            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            last_improve = checkpoint['last_improve']
            best_dev_micro_f1 = checkpoint['best_dev_micro_f1']
            
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
        else:
            global_step = 0
            last_improve = 0
            start_epoch = 0
            best_dev_micro_f1 = 0.0

        early_stop = False
        total_step = len(train_loader) * self.args.epochs
        self.model.train()
        for epoch in range(start_epoch, self.args.epochs):
            for data in train_loader:
                global_step += 1
                input_ids = data[0].to(self.device)
                attention_mask = data[1].to(self.device)
                labels = data[2].to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if global_step % 10 == 0:
                    logger.info(
                        "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                self.writer.add_scalar("train_loss", loss.item(), global_step)
                
                if global_step % self.args.eval_step == 0:
                    dev_loss, dev_pred, dev_target = self.dev(dev_loader)
         
                
                    accuracy, micro_f1, macro_f1 = self.get_metrics(dev_pred, dev_target)
                    logger.info(
                        "【dev】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(dev_loss, accuracy,
                                                                                                   micro_f1, macro_f1))
                    self.writer.add_scalar("dev_loss", dev_loss, global_step)
                    self.writer.add_scalar("dev_acc", accuracy, global_step)
                    self.writer.add_scalar("dev_micro_f1", micro_f1, global_step)
                    self.writer.add_scalar("dev_macro_f1", macro_f1, global_step)

                    if micro_f1 > best_dev_micro_f1:
                        last_improve = global_step
                        best_dev_micro_f1 = micro_f1
                        logger.info("保存当前最优模型")
                        checkpoint = {
                            'epoch': epoch,
                            'global_step':global_step,
                            'last_improve':last_improve,
                            'best_dev_micro_f1':best_dev_micro_f1,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        checkpoint_path = self.args.checkpoint_path
                        best_model_state_path = self.args.best_model_state_path
                        torch.save(checkpoint, checkpoint_path)
                        torch.save(self.model.state_dict(), best_model_state_path)
                        
                if global_step - last_improve > self.args.early_stop_patience:
                    early_stop = True
                    break
            if early_stop:
                logger.info("no improve for long time, early stop!!")
                break
        self.writer.close()
                
    
    def dev(self, dev_loader):
        self.model.eval()
        pred, target = [], []
        total_loss = 0.0
        with torch.no_grad():
            for data in dev_loader:
                input_ids = data[0].to(self.device)
                attention_mask = data[1].to(self.device)
                labels = data[2].to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                
                outputs = torch.sigmoid(logits).cpu().detach().numpy().tolist() #TODO
                outputs = (np.array(outputs) > 0.6).astype(int)
                pred.extend(outputs.tolist())
                target.extend(labels.cpu().detach().numpy().tolist())
        
        self.model.train()
        return total_loss/len(dev_loader), pred, target

    def test(self, test_loader):
        self.model.load_state_dict(torch.load(self.args.best_model_state_path))
        _, pred, target = self.dev(test_loader)
        
        accuracy, micro_f1, macro_f1 = self.get_metrics(pred, target)
        logger.info("【test】 accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(accuracy, micro_f1, macro_f1))
        report = self.get_classification_report(pred, target, self.labels)
        logger.info(report)



    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, micro_f1, macro_f1



    def get_classification_report(self, outputs, targets, labels):
        # confusion_matrix = multilabel_confusion_matrix(targets, outputs)
        report = classification_report(targets, outputs, target_names=labels)
        return report

        



def main():
    config = get_config()
    utils.set_logger(os.path.join(config.log_path, 'main.log'))
    utils.set_seed(config.seed)
    train_loader = gen_dataloader(config, read_json_data, mode='train')
    dev_loader = gen_dataloader(config, read_json_data, mode='dev')
    test_loader = gen_dataloader(config, read_json_data, mode='test')

    trainer = Trainer(config)
    logger.info("start training")

    trainer.train(train_loader, dev_loader)

    ## 继续训练
    # trainer.train(train_loader, dev_loader, checkpoint_path="./model_checkpoint/ckpt.pt")
    
    ## 测试
    trainer.test(dev_loader)
    # trainer.test(test_loader)


if __name__ == "__main__":
    main()