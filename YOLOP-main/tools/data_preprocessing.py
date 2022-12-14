import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from lib.config import cfg
import torchvision.transforms as transforms
import lib.dataset as dataset
import json
from lib.dataset import BddDataset
import numpy
import pickle 
from contextlib import redirect_stdout
import yaml
from lib.dataset import AutoDriveDataset

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():


    print("begin to load data")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    

    with open('./lib/dataset/train_dataset_cfg.yml', 'w') as f:
        with redirect_stdout(f): print(train_dataset.cfg.dump())
    
    with open('./lib/dataset/train_dataset_bdd.json', 'w') as fout:
        json.dump(train_dataset.db , fout, cls=NumpyArrayEncoder)

    with open("./lib/dataset/train_dataset_cfg.yml", 'r') as stream:
       train_dataset_cfg = yaml.safe_load(stream)
    
    with open('./lib/dataset/train_dataset_bdd.json', "r") as read_file:
        train_dataset_db = json.load(read_file)


    train_dataset_ld = BddDataset(cfg=train_dataset_cfg,
            is_train=True,
            inputsize=train_dataset_cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    ).set_db(train_dataset_db)
    print(train_dataset_ld)


    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    
    with open('./lib/dataset/valid_dataset_cfg.yml', 'w') as f:
        with redirect_stdout(f): print(valid_dataset.cfg.dump())
    
    with open('./lib/dataset/valid_dataset_bdd.json', 'w') as fout:
        json.dump(valid_dataset.db , fout, cls=NumpyArrayEncoder)



if __name__ == '__main__':
    main()