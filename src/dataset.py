import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.ndimage as ndimage

class NoduleDataset(Dataset):
    def __init__(self, nodule_dir, labels_file, patient_ids, augmentation = False, difficulty = "none"):
        self.nodule_dir = nodule_dir
        self.labels_df = pd.read_csv(labels_file)
        self.labels_df = self.labels_df[self.labels_df['patient_id'].isin(patient_ids)]

        if difficulty == "easy":
            self.labels_df = self.labels_df[self.labels_df['malignancy'].isin([1, 5])]
        elif difficulty == "hard":
            self.labels_df = self.labels_df[self.labels_df['malignancy'].isin([2, 4])]
        
        self.nodules = self.labels_df['nodule_id'].values
        self.labels = self.labels_df['malignancy_truth'].values

        
        self.augmentation = augmentation 

        self.angles = [45, 90, 135, 180, 225, 270, 315]
        self.rotation_dimensions = [(x, 0, 0) for x in self.angles] + [(0, y, 0) for y in self.angles] + [(0, 0, z) for z in self.angles]

    def __len__(self):
        return len(self.nodules) * (len(self.rotation_dimensions) if self.augmentation else 1)
    
    def __getitem__(self, index):
        
        original_index = index // (len(self.rotation_dimensions) if self.augmentation else 1)
        rotation_index = index % (len(self.rotation_dimensions) if self.augmentation else 1)

        nodule_id = self.nodules[original_index]
        nodule_path = os.path.join(self.nodule_dir, f"{nodule_id}.npy")
        nodule = np.load(nodule_path).astype(np.float32)

        if self.augmentation:
            x, y, z = self.rotation_dimensions[rotation_index]
            nodule = self.rotate(nodule, x, y, z)

        nodule = torch.from_numpy(nodule).unsqueeze(0)
        label = self.labels[original_index]
        return nodule, torch.tensor(label,dtype=torch.float32)
    
    def rotate(self, volume, x, y, z):
        if x:
            volume = ndimage.rotate(volume, x, axes=(1, 2), reshape=False, mode="nearest")
        elif y:
            volume = ndimage.rotate(volume, y, axes=(0, 2), reshape=False, mode="nearest")
        elif z:
            volume = ndimage.rotate(volume, z, axes=(0, 1), reshape=False, mode="nearest")
        return volume
    
