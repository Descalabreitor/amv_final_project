import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from lib.config import cfg
import torchvision.transforms as transforms
import lib.dataset as dataset
import json

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

    with open("./lib/dataset/train_dataset.json", "w") as final:
        json.dump(train_dataset, final)

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    
    with open("./lib/dataset/valid_dataset.json", "w") as final:
        json.dump(valid_dataset, final)

if __name__ == '__main__':
    main()