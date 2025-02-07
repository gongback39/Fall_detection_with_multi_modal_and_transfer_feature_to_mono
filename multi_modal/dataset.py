import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class make_dataset(Dataset):
    def __init__(self, dir, data_types, transform = None):
        self.img_dir = []
        self.sensor_dir = []
        self.labels = []
        self.ratio = []
        self.data_types = data_types
        self.transform = transform

        self.body_parts = ['Pelvis', 'Head', 'Left Forearm', 'Left Lower Leg', 'Left Shoulder',
                           'Left Upper Arm', 'Left Upper Leg', 'Right Forearm', 'Right Lower Leg',
                           'Right Shoulder', 'Right Upper Arm','Right Upper Leg']

        dir = dir + '/이미지'
        subdir = ['/N/N', '/Y/BY', '/Y/FY', '/Y/SY']
        
        for i in range(len(subdir)):
            self.get_imgs(dir+subdir[i],  label = 0 if i == 0 else 1)
        
    def get_imgs(self, dir, label):
        self.ratio.append(len(os.listdir(dir)))
        for imgs in os.listdir(dir):
            imgs = os.path.join(dir, imgs)
            self.get_sensors(imgs)
            images = []
            for image in sorted(os.listdir(imgs)):
                image = os.path.join(imgs, image)
                images.append(image)
            self.img_dir.append(images)
            self.labels.append(label)
    
    def get_sensors(self, dir):
        sensors = dir.replace("이미지", "센서")
        self.sensor_dir.append(os.path.join(sensors, os.listdir(sensors)[0]))
            
    
    def __len__(self):
        return len(self.img_dir)
    
    def __getitem__(self, idx):
        images = self.img_dir[idx]
        images = [Image.open(image).convert("RGB") for image in images]
        images = [self.transform(image) for image in images]
        
        images_tensor = torch.stack(images) 
        images_tensor = images_tensor.permute(1, 0, 2, 3)

        sensors = self.sensor_dir[idx]
        df = pd.read_csv(sensors)
        body_part_tensors = []
        for body_part in self.body_parts:
            part_cols = []
            for data_type in self.data_types:
                part_cols += [col for col in df.columns if data_type in col and body_part in col]
            part_data = df[part_cols].values
            part_data_avg = part_data.reshape(10, 60, -1).mean(axis=1)
            body_part_tensor = torch.tensor(part_data_avg, dtype=torch.float32)
            body_part_tensors.append(body_part_tensor)
        sensors_tensor = torch.stack(body_part_tensors)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return images_tensor, sensors_tensor, label

def create_dataloaders(dir, batch_size=4, test_ratio=0.2, image_size=640, workers = 1):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = make_dataset(dir=dir, data_types = ['Segment Acceleration', 'Segment Angular Velocity', 'Sensor Magnetic Field'], transform=transform)

    if test_ratio == 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    else:
        n_idx = list(range(dataset.ratio[0]))
        backward_idx = list(range(dataset.ratio[0], sum(dataset.ratio[:2])))
        forward_idx = list(range(sum(dataset.ratio[:2]), sum(dataset.ratio[:3])))
        sideways_idx = list(range(sum(dataset.ratio[:3]), sum(dataset.ratio[:4])))

        n_train_idx, n_val_idx = train_test_split(n_idx, test_size=test_ratio, random_state=42)
        backward_train_idx, backward_val_idx = train_test_split(backward_idx, test_size=test_ratio, random_state=42)
        forward_train_idx, forward_val_idx = train_test_split(forward_idx, test_size=test_ratio, random_state=42)
        sideways_train_idx, sideways_val_idx = train_test_split(sideways_idx, test_size=test_ratio, random_state=42)

        train_idx = np.concatenate([n_train_idx, backward_train_idx, forward_train_idx, sideways_train_idx])
        val_idx = np.concatenate([n_val_idx, backward_val_idx, forward_val_idx, sideways_val_idx])

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

        return train_dataloader, val_dataloader