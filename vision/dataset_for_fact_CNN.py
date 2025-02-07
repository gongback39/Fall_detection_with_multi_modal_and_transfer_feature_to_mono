import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class video_dataset(Dataset):
    def __init__(self, dir, transform = None):
        self.inputs = []
        self.labels = []
        self.ratio = []

        self.transform = transform
        
        negative_dir = dir + '/N/N'
        positive_dir = dir + '/Y'
        sub_dirs_in_Y = sorted(os.listdir(positive_dir))

        self.get_videos(negative_dir, label=0)
        for sub_dir in sub_dirs_in_Y:
            self.get_videos(positive_dir+ "/" + sub_dir, label=1)
    
    def get_videos(self, dir, label):
        self.ratio.append(len(os.listdir(dir)))
        for videos in os.listdir(dir):
            videos = os.path.join(dir, videos)
            images = []
            for image in sorted(os.listdir(videos)):
                image = os.path.join(videos, image)
                images.append(image)
            self.inputs.append(images)
            self.labels.append(label)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        images = self.inputs[idx]
        images = [Image.open(image).convert("RGB") for image in images]
        images = [self.transform(image) for image in images]
        
        images_tensor = torch.stack(images) 
        images_tensor = images_tensor.permute(1, 0, 2, 3)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return images_tensor, label

def create_dataloaders(dir, batch_size=4, test_ratio = 0.2, image_size = 640, workers = 1):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = video_dataset(dir=dir, transform=transform)

    if test_ratio == 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    else :
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