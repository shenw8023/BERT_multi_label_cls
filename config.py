
import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_path', default='./bert_pretrain_model/bert-base-chinese',
                        help='bert dir')
    parser.add_argument('--vocab_path', default='./bert_pretrain_model/bert-base-chinese/vocab.txt')
    parser.add_argument('--train_path', default='./data/train.json',
                        help='data dir')
    parser.add_argument('--dev_path', default='./data/dev.json')
    parser.add_argument('--test_path', default='./data/dev.json')  ## TODO 暂无
    parser.add_argument('--label_path', default='./data/labels.txt')
    # parser.add_argument('--model_save_path', type=str, default='./model_checkpoint/')
    parser.add_argument('--checkpoint_path', default="./model_checkpoint/ckpt.pt")
    parser.add_argument('--best_model_state_path', default='./model_checkpoint/best_model.pt')
    
    parser.add_argument('--max_seq_len', default=128)
    parser.add_argument('--batch_size', default=32)

    parser.add_argument('--bert_dropout_prob', default=0.1)

    parser.add_argument('--device_id', type=int, default=0, choices=[-1, 0, 1, 2], help='-1使用cpu，其他表示gpu_id')
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_step', type=int, default=200)
    parser.add_argument('--early_stop_patience', type=int, default=4000, help='多少个batch没有提升，进行早停') #11958/32=370
    parser.add_argument('--log_path', type=str, default='./logs/')

    parser.add_argument('--seed', default=123)


    args = parser.parse_args()
    return args