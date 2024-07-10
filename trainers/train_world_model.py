import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import numpy as np
import os
import argparse
import time
import traceback

from networks import World_Model
from dataset import VideoDataset
from utils import Logger, setup_seed, draw_figure, convert_seconds, normalize_image, unnormalize_image

# 3407 is all you need
setup_seed(3407)
fmt = "----- {:^25} -----"

class WM_Trainer: 
    def __init__(self, args):
        # Set hyper-parameters
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.dataset_path = args.dataset_path
        self.save_path = args.save_path
        self.it_model_path = args.it_model_path
        self.wm_model_path = args.wm_model_path

        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim

        # Set device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Create neural network
        self.model = self.create_neural_network()

        # Define optimizer
        self.optim = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)

        # Load dataset
        self.train_loader, self.val_loader = self.load_dataset()


    def run(self): 
        print(fmt.format("Start training") + '\n')
        min_loss = -1
        best_epoch = 0
        epoch_list, loss_list, val_list = [], [], []
        very_start = time.time()
        formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(very_start))
        self.log = Logger(f"logs/WM-{formatted_time}.txt")

        try:
            for epoch in range(self.epochs):
                start = time.time()

                # Train one epoch
                current_loss = self.train(epoch)
                val_loss = self.val()

                # Save model
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "current.pth"))
                if current_loss < min_loss or min_loss == -1:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, "best.pth"))
                    self.log("Update the best model")
                    min_loss = current_loss
                    best_epoch = epoch + 1

                # Draw figure
                epoch_list.append(epoch + 1)
                loss_list.append(current_loss)
                val_list.append(val_loss)
                draw_figure(epoch_list, loss_list, "Loss", f"./logs/WM-train-{formatted_time}.png")
                draw_figure(epoch_list, val_list, "Loss", f"./logs/WM-val-{formatted_time}.png")

                # Elapsed time
                end = time.time()
                use_time = int(end - start)
                self.log(f"Elapsed time: {use_time // 60}m {use_time % 60}s\n")
            
        except Exception as e:
            self.log(traceback.format_exc())
        
        except KeyboardInterrupt: 
            print()
        
        very_end = time.time()
        total_time = int(very_end - very_start)

        self.log(f"Training finished! Total elapsed time: {convert_seconds(total_time)}, "
            f"Best Epoch: {best_epoch}, Min Loss: {min_loss:.4f}")
        
        
    def train(self, epoch_num, count=1000):
        self.model.train()
        running_loss, total_loss, total = 0.0, 0.0, 0
        self.log(f"< Epoch {epoch_num + 1} >")
        for batch_idx, data in enumerate(self.train_loader, 0):
            # Get data
            frame_list, action_list = data
            frame_list, action_list = frame_list.to(self.device), action_list.to(self.device)

            self.optim.zero_grad()

            # Forward + Backward + Updaate
            loss = self.model(frame_list, action_list)
            loss = loss.sum()
            loss.backward()
            self.optim.step()

            # Calculate loss
            running_loss += loss.item()
            total_loss += loss.item()
            if batch_idx % count == count - 1:
                print('\r', end='')
                self.log('Batch %d   \t loss: %.6f' % (batch_idx + 1, running_loss / count))
                running_loss = 0.0
            else: 
                print(f"\r[{batch_idx % count + 1} / {count}]", end='')
            total += 1
        print('\r', end='')

        return total_loss / total
    
    def val(self):
        self.model.eval()
        total_loss, total = 0.0, 0
        with torch.no_grad(): 
            for data in self.val_loader: 
                frame_list, action_list = data
                frame_list, action_list = frame_list.to(self.device), action_list.to(self.device)

                # Forward only
                loss = self.model(frame_list, action_list)
                loss = loss.sum()

                # Calculate loss
                total_loss += loss.item()
                total += 1

                print(f"\r[{total} / {len(self.val_loader)}]", end='')
            print('\r', end='')
        val_loss = total_loss / total
        self.log(f"Validation\tLoss = {val_loss: .4f}")

        return val_loss

        
    def create_neural_network(self): 
        print(fmt.format("Create neural network"))
        model = World_Model(vocab_size=self.vocab_size, embed_dim=self.embed_dim, 
                            num_frames=7, num_image_tokens=576, num_action_tokens=1).to(self.device)

        # Load pretrained model or create a new model
        if self.wm_model_path != '':
            print(f"Loading pretrained model: {self.wm_model_path}")
            model.load_state_dict(torch.load(self.wm_model_path))
        else:
            print("Creating new world model")
            print(f"Loading pretrained model for image_tokenizer: {self.it_model_path}")
            model.load_image_tokenizer(self.it_model_path)

        # device_count = torch.cuda.device_count()
        # print(f"Using {device_count} GPUs")

        # model = nn.DataParallel(model, device_ids=[i for i in range(device_count)])
        print()

        return model
    
    def load_dataset(self): 
        print(fmt.format("Load dataset"))
        train_set = VideoDataset(root_path=os.path.join(self.dataset_path, "train"), num_frame=7)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        val_set = VideoDataset(root_path=os.path.join(self.dataset_path, "val"), num_frame=7)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        print()
        return train_loader, val_loader
        

if __name__ == "__main__":
    fmt = "----- {:^25} -----"
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm-model-path', type=str, default='')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--dataset-path', type=str, default='/root/share/minecraft')
    parser.add_argument('--save-path', type=str, default='../weights/')
    parser.add_argument('--it-model-path', type=str, default='../weights/image_tokenizer/0608.pth')
    args = parser.parse_args()

    trainer = WM_Trainer(args=args)
    trainer.run()
