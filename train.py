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

from networks import Image_Tokenizer, Discriminator
from dataset import NetDataset
from utils import Logger, setup_seed, draw_figure, convert_seconds, normalize_image, unnormalize_image

# 3407 is all you need
setup_seed(3407)

def train(epoch_num, count=10):
    model.train()
    running_loss, total_loss, total = 0.0, 0.0, 1
    log(f"< Epoch {epoch_num + 1} >")
    for batch_idx, data in enumerate(train_loader, 0):
        # Get data
        inputs = data
        inputs = inputs.to(device)

        optim_G.zero_grad()
        optim_D.zero_grad()

        # Forward + Backward + Update
        fake = model.module.generate(inputs)
        real = inputs

        # Discriminate 
        dis_real, dis_fake = discriminator(real), discriminator(fake.detach())
        real_loss, fake_loss = F.binary_cross_entropy(dis_real, torch.ones_like(dis_real)), F.binary_cross_entropy(dis_fake, torch.zeros_like(dis_fake))
        dis_loss = real_loss + fake_loss

        # Optimize discriminator
        dis_loss.backward()
        optim_D.step()

        # Calculate tokenizer loss
        perceptual_loss, vq_loss, l1_loss, l2_loss, outputs = model(inputs)
        dis_gen = discriminator(outputs)
        gan_loss = F.binary_cross_entropy(dis_gen, torch.ones_like(dis_gen))
        loss = 0.2 * l1_loss + 2.0 * l2_loss + 0.1 * perceptual_loss + 1.0 * gan_loss + 1.0 * vq_loss
        loss = loss.sum()
        loss.backward()
        optim_G.step()

        # Sum loss
        running_loss += loss.item()
        total_loss += loss.item()
        if batch_idx % count == count - 1:
            print('\r', end='')
            log('Batch %d\t loss: %.6f' % (batch_idx + 1, running_loss / count))
            running_loss = 0.0
        else: 
            print(f"\r[{batch_idx % count + 1} / {count}]", end='')
        total += 1
    print('\r', end='')

    return total_loss / total

def val(epoch_num):
    # Load image
    image_path = "/root/share/cat/train/image/00850486-IMG_20211020_091815.jpg"
    image = cv2.imread(image_path)

    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image)
    image = np.transpose(image, (2, 0, 1))

    data = np.array(image).astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(0).to(device)

    # Generate and save
    model.eval()
    with torch.no_grad():
        sample = model.module.generate(data)
    image = sample[0].permute(1, 2, 0).cpu().detach().numpy()
    image_save = unnormalize_image(image).astype('uint8')
    image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join("outputs", f"epoch {epoch_num + 1}.jpg"), image_save)


if __name__ == "__main__":
    fmt = "----- {:^25} -----"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--dataset-path', type=str, default='/root/share/cat/train/image')
    args = parser.parse_args()

    # Set hyper-parameters
    epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dataset_path = args.dataset_path
    model_path = args.model_path

    # Set device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Create neural network
    print(fmt.format("Create neural network"))
    model = Image_Tokenizer()
    discriminator = Discriminator(patch=False)

    device_count = torch.cuda.device_count()
    print(f"Using {device_count} GPUs")

    # Load pretrained model or create a new model
    if model_path != '':
        print(f"Loading pretrained model: {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Creating new model")

    model = model.to(device)
    discriminator = discriminator.to(device)
    model = nn.DataParallel(model, device_ids=[i for i in range(device_count)])
    discriminator = nn.DataParallel(discriminator, device_ids=[i for i in range(device_count)])
    print()

    # Define optimizer
    optim_G = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.01)
    optim_D = optim.Adam(params=discriminator.parameters(), lr=learning_rate, weight_decay=0.01)

    # Load dataset
    print(fmt.format("Loading dataset"))
    train_set = NetDataset(folder=dataset_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    print()

    # Start training
    print(fmt.format("Start training") + '\n')
    min_loss = -1
    best_epoch = 0
    epoch_list, loss_list = [], []
    very_start = time.time()
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(very_start))
    log = Logger(f"logs/{formatted_time}.txt")

    try:
        for epoch in range(epochs):
            start = time.time()

            # Train one epoch
            current_loss = train(epoch)
            val(epoch)

            # Save model
            torch.save(model.state_dict(), f"./weights/current.pth")
            if current_loss < min_loss or min_loss == -1:
                torch.save(model.state_dict(), f"./weights/best.pth")
                log("Update the best model")
                min_loss = current_loss
                best_epoch = epoch + 1

            # Draw figure
            epoch_list.append(epoch + 1)
            loss_list.append(current_loss)
            draw_figure(epoch_list, loss_list, "Loss", f"./logs/{formatted_time}.png")

            # Elapsed time
            end = time.time()
            use_time = int(end - start)
            log(f"Elapsed time: {use_time // 60}m {use_time % 60}s\n")
        
    except Exception as e:
        log(traceback.format_exc())
    
    except KeyboardInterrupt: 
        print()
    
    very_end = time.time()
    total_time = int(very_end - very_start)

    log(f"Training finished! Total elapsed time: {convert_seconds(total_time)}, "
        f"Best Epoch: {best_epoch}, Min Loss: {min_loss:.4f}")