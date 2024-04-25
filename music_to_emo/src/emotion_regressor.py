import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def r_squared(predictions, targets):
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

class MinMaxScaler:
    def __init__(self):
        self.min = 0.
        self.max = 255.

    def fit(self, x):
        self.min_val = torch.min(x, dim=0)[0] 
        self.max_val = torch.max(x, dim=0)[0]

    def transform(self, x):
        return (x - self.min) / (self.max - self.min + 1e-6)

    def inverse_transform(self, x):
        return x * (self.max - self.min) + self.min

class EmotionRegressor(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.scaler = MinMaxScaler()

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=4,dropout=0.25, batch_first=True)

        self.fc_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(8, 2),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        x = self.scaler.transform(x)
        x, _ = self.lstm(x)
        #fully connected layers applied for each time step
        x = x.contiguous().view(-1, x.shape[2])
        x = self.fc_layers(x)
        #reshape to original shape
        x = x.view(batch_size, seq_length, -1)
        return x
    
    def training_step(self, batch, batch_idx):
        features, valence_labels, arousal_labels = batch

        preds = self(features)

        valence_hat = preds[:, :, 0]
        arousal_hat = preds[:, :, 1]

        valence_loss = F.mse_loss(valence_hat, valence_labels)
        arousal_loss = F.mse_loss(arousal_hat, arousal_labels)
        total_loss = valence_loss + arousal_loss

        # Logging the losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_valence_loss', valence_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_arousal_loss', arousal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {'valence_loss': valence_loss, 'arousal_loss': arousal_loss, 'loss': total_loss}
    
    def validation_step(self, batch, batch_idx):
        features, valence_labels, arousal_labels = batch

        preds = self(features)

        valence_hat = preds[:, :, 0]
        arousal_hat = preds[:, :, 1]

        valence_loss = F.mse_loss(valence_hat, valence_labels)
        arousal_loss = F.mse_loss(arousal_hat, arousal_labels)
        total_loss = valence_loss + arousal_loss

        #r squared
        valence_r2 = r_squared(valence_hat, valence_labels)
        arousal_r2 = r_squared(arousal_hat, arousal_labels)

        # Logging the losses
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_valence_loss', valence_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_arousal_loss', arousal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_valence_r2', valence_r2, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_arousal_r2', arousal_r2, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {'valence_loss': valence_loss, 'arousal_loss': arousal_loss, 'loss': total_loss, 'valence_r2': valence_r2, 'arousal_r2': arousal_r2}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
