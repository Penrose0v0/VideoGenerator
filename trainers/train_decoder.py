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

from networks import Decoder, World_Model
from dataset import ImageDataset
from utils import Logger, setup_seed, draw_figure, convert_seconds, normalize_image, unnormalize_image

# 3407 is all you need
setup_seed(3407)
fmt = "----- {:^25} -----"

class VD_Trainer: 
    def __init__(self, args):
        # Set hyper-parameters
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.dataset_path = args.dataset_path
        self.save_path = args.save_path
        self.it_model_path = args.it_model_path
        self.vd_model_path = args.vd_model_path
        self.wm_model_path = args.wm_model_path

        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim

        # Set device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Create neural network
        self.model, self.wm = self.create_neural_network()

        # Define optimizer
        self.optim = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)

        # Load dataset
        self.train_loader = self.load_dataset()


    def run(self): 
        print(fmt.format("Start training") + '\n')
        min_loss = -1
        best_epoch = 0
        epoch_list, loss_list = [], []
        very_start = time.time()
        formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(very_start))
        self.log = Logger(f"logs/{formatted_time}.txt")

        try:
            for epoch in range(self.epochs):
                start = time.time()

                # Train one epoch
                current_loss = self.train(epoch)
                self.val(epoch)

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
                draw_figure(epoch_list, loss_list, "Loss", f"./logs/{formatted_time}.png")

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
        running_loss, total_loss, total = 0.0, 0.0, 1
        self.log(f"< Epoch {epoch_num + 1} >")
        for batch_idx, data in enumerate(self.train_loader, 0):
            # Get data
            inputs = data
            inputs = inputs.to(self.device)

            self.optim.zero_grad()

            # Forward + Backward + Updaate
            loss = self.model(inputs)
            loss = loss.sum()
            loss.backward()
            self.optim.step()

            # Sum loss
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
    
    def val(self, epoch_num):
        # Generate action (only 'forward')
        action = [0 for _ in range(25)]
        action[6] = 1
        action = torch.from_numpy(np.array(action)).unsqueeze(0).repeat(5, 1).unsqueeze(0).to(self.device)

        # Load image
        image_path = "/root/share/minecraft/val/sample_10/0.png"
        image = cv2.imread(image_path)

        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = normalize_image(image)
        image = np.transpose(image, (2, 0, 1))

        data = np.array(image).astype(np.float32)
        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(self.device)

        # Generate and save
        self.model.eval()
        with torch.no_grad():
            generated_tokens_list = self.wm.predict(frame_list=data, action_list=action)
            generated_tokens = torch.stack(generated_tokens_list).squeeze()
            decoded = self.model.decode_tokens(generated_tokens)
        for i in range(decoded.shape[0]): 
            image = decoded[i].permute(1, 2, 0).cpu().detach().numpy()
            image_save = unnormalize_image(image).astype('uint8')
            image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join("outputs", f"{epoch_num + 1}-{i + 1}.jpg"), image_save)

        
    def create_neural_network(self): 
        print(fmt.format("Create neural network"))
        model = Decoder(vocab_size=self.vocab_size, embed_dim=self.embed_dim).to(self.device)
        wm = World_Model(vocab_size=self.vocab_size, embed_dim=self.embed_dim, 
                         num_frames=7, num_image_tokens=576, num_action_tokens=25).to(self.device)

        # Load pretrained model or create a new model
        if self.vd_model_path != '':
            print(f"Loading pretrained model: {self.vd_model_path}")
            model.load_state_dict(torch.load(self.vd_model_path))
        else:
            print("Creating new video decoder")
            print(f"Loading pretrained model for image_tokenizer: {self.it_model_path}")
            model.load_image_tokenizer(self.it_model_path)
        print(f"Loading pretrained model for world model: {self.wm_model_path}")
        wm.load_state_dict(torch.load(self.wm_model_path))

        # device_count = torch.cuda.device_count()
        # print(f"Using {device_count} GPUs")

        # model = nn.DataParallel(model, device_ids=[i for i in range(device_count)])
        print()

        return model, wm
    
    def load_dataset(self): 
        print(fmt.format("Load dataset"))
        train_set = ImageDataset(root_path=self.dataset_path)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        print()
        return train_loader
        

if __name__ == "__main__":
    fmt = "----- {:^25} -----"
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaia-model-path', type=str, default='')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--dataset-path', type=str, default='/root/share/minecraft')
    parser.add_argument('--save-path', type=str, default='./weights/')
    parser.add_argument('--it-model-path', type=str, default='./weights/image_tokenizer/0608.pth')
    args = parser.parse_args()

    trainer = VD_Trainer(args=args)
    trainer.run()
