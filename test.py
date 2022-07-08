
from dataset import read_json_data, gen_dataloader, load_dataset
from config import get_config


def test_dataloader():
    config = get_config()
    
    # inputs, labels = load_dataset(config, read_json_data, mode='dev')
    # print(labels.shape)
    # for i , j in inputs.items():
    #     print(j.shape)

    dl_dev = gen_dataloader(config, read_json_data, mode='dev')
    for i, j, k in dl_dev:
        print(i.shape)
        print(j.shape)
        print(k.shape)
        print(i)
        break
    
test_dataloader()