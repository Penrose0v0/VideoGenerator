import torch
from torch.utils.data.dataset import Dataset
import os
import cv2
import numpy as np

from utils import normalize_image

class ImageDataset(Dataset):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.input_shape = (512, 288)
        self.data = self.read_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = np.array(self.data[item]).astype(np.float32)
        return torch.from_numpy(data)

    @property
    def read_data(self):
        print(f"Dataset Path: {self.folder}")
        files = os.listdir(self.folder)

        data = []
        total = 0
        for file in files:
            # Add image and preprocess
            image = cv2.imread(os.path.join(self.folder, file))
            image = cv2.resize(image, self.input_shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(normalize_image(image), (2, 0, 1))

            data.append(image)
            total += 1
            print(f"\rReading data... [{total} / {len(files)}] ", end='')

        print("Completed! ")
        return data
    
class VideoDataset(Dataset): 
    def __init__(self, root_path, num_frame=7):
        super().__init__()
        self.root_path = root_path
        self.input_shape = (512, 288)
        self.num_frame = num_frame

        self.data = self.read_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def read_data(self):
        print(f"Dataset Path: {self.root_path}")
        scenes = os.listdir(self.root_path)

        data = []
        total = 0
        pre_process = lambda x: np.transpose(normalize_image(cv2.cvtColor(
            cv2.resize(x, self.input_shape), cv2.COLOR_BGR2RGB)), (2, 0, 1))
        for scene in scenes: 
            current_path = os.path.join(self.root_path, scene)
            with open(os.path.join(current_path, "actions.txt"), 'r', encoding='utf-8') as file: 
                actions = file.readlines()
            # frames = os.listdir(current_path)

            frame_list = [pre_process(cv2.imread(os.path.join(current_path, "0.png")))]
            action_list = []
            for i in range(len(actions)): 
                # Get current and next frame, then pre-process
                current_frame = pre_process(cv2.imread(os.path.join(current_path, f"{i+1}.png")))
                # next_frame = cv2.imread(os.path.join(current_path, f"{i+1}.png"))
                # current_frame, next_frame = map(pre_process, (current_frame, next_frame))
                
                # Get action and convert
                next_action = list(map(eval, actions[i].strip().strip('|').split('|')))
                next_action = self.act_to_np(next_action)

                frame_list.append(current_frame)
                action_list.append(next_action)
                assert len(frame_list) == len(action_list) + 1

                # Add new datum to data list
                if len(frame_list) == self.num_frame:
                    datum = (np.array(frame_list), np.array(action_list))
                    data.append(datum) 
                    frame_list.pop(0)
                    action_list.pop(0)
                    total += 1
                    print(f"\rReading data... [{total} / {(len(actions) - self.num_frame + 1) * len(scenes)}] ", end='')
                    if total == 2:
                        break
            break
        print("Completed! ")
        return data

    def act_to_np(self, action): 
        act_tensor = []
        for act in action: 
            if not isinstance(act, int): 
                act_tensor.append(act[0] / 180)
                act_tensor.append(act[1] / 180)
            else: 
                act_tensor.append(act)
        # return torch.tensor(act_tensor, dtype=torch.float)
        return np.array(act_tensor)
    
    
if __name__ == "__main__": 
    dataset = VideoDataset("/root/share/minecraft")
    # print(dataset[0][0])
    # print(dataset[1][0])
    print(len(dataset[0][0]))
    print(len(dataset[0][1]))
    print(len(dataset[1][1]))