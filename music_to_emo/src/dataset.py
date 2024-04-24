import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import logging

def preprocess_and_save(audio_dir, valence_path, arousal_path, output_dir):
    #instantiate the model
    vggish = torch.hub.load('harritaylor/torchvggish', 'vggish').eval()

    # Load the labels
    valence = pd.read_csv(valence_path, index_col=0)
    arousal = pd.read_csv(arousal_path, index_col=0)
    valence.columns = valence.columns.str[7:-2].astype(int)-15000
    arousal.columns = arousal.columns.str[7:-2].astype(int)-15000

    files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

    for filename in files:
        full_path = os.path.join(audio_dir, filename)
        track_id = int(os.path.basename(filename)[:-4])
        features = vggish(full_path).detach()  # Ensure no gradients are attached

        #make output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save features
        torch.save(features, os.path.join(output_dir, f"{track_id}_features.pt"))
        
        #get nearest times for labels largest offset is approximately 0.25s or 250ms
        feature_times = np.linspace(0, features.shape[0]*960, len(features), endpoint=False)
        nearest_times = np.array(valence.columns)[np.abs(np.array(valence.columns)[:, None] - feature_times).argmin(axis=0)]
        valence_labels = valence.loc[track_id, nearest_times].values
        arousal_labels = arousal.loc[track_id, nearest_times].values

        torch.save(torch.tensor(valence_labels, dtype=torch.float32), os.path.join(output_dir, f"{track_id}_valence.pt"))
        torch.save(torch.tensor(arousal_labels, dtype=torch.float32), os.path.join(output_dir, f"{track_id}_arousal.pt"))


class ProcessedDEAMDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('_features.pt')]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        base_name = self.files[idx].replace('_features.pt', '')
        
        features = torch.load(os.path.join(self.data_dir, f"{base_name}_features.pt"))
        valence_labels = torch.load(os.path.join(self.data_dir, f"{base_name}_valence.pt"))
        arousal_labels = torch.load(os.path.join(self.data_dir, f"{base_name}_arousal.pt"))

        if(features.shape[0] != 31):
            logging.warning(f"Features for {base_name} have a shape of {features.shape[0]}")
        
        return features, valence_labels, arousal_labels
    

class DEAMDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, transform=None, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def setup(self, stage=None):
        #initialize full dataset
        self.full_dataset = ProcessedDEAMDataset(self.data_dir)

        #create train and val datasets
        self.train_size = int(0.8 * len(self.full_dataset))
        self.val_size = len(self.full_dataset) - self.train_size
        self.train_dataset, self.val_dataset = random_split(self.full_dataset, [self.train_size, self.val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)