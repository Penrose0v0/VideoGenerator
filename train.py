import argparse

from trainers import *
from utilities import setup_seed

# 3407 is all you need
setup_seed(3407)

if __name__ == "__main__":
    fmt = "----- {:^25} -----"
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-mode', type=str, default='wm')

    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)

    parser.add_argument('--vocab-size', type=float, default=4096)
    parser.add_argument('--embed-dim', type=float, default=2048)

    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--wm-model-path', type=str, default='')
    parser.add_argument('--it-model-path', type=str, default='./weights/image_tokenizer/0724.pth')
    parser.add_argument('--vd-model-path', type=str, default='')
    parser.add_argument('--dataset-path', type=str, default='/root/share/minecraft')
    parser.add_argument('--save-path', type=str, default='./weights/world_model')

    args = parser.parse_args()

    train_mode = args.train_mode

    if train_mode == "it": 
        trainer = IT_Trainer(args=args)

    if train_mode == "wm": 
        trainer = WM_Trainer(args=args)

    if train_mode == "vd": 
        trainer = VD_Trainer(args=args)

    trainer.run()
