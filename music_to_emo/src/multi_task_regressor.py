import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def r_squared(predictions, targets):
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

class MultiTaskRegressor(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=4,dropout=0.25, batch_first=True)
        self.relu = nn.ReLU()
        # Separate FC layers for valence
        self.fc_valence = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()  # Assuming valence is a score between -1 and 1
        )
        
        # Separate FC layers for arousal
        self.fc_arousal = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()  # Assuming arousal is a score between -1 and 1
        )


    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        x, _ = self.lstm(x)

        #fully connected layers applied for each time step
        x = x.contiguous().view(-1, x.shape[2])
        x_valence = self.fc_valence(x).view(batch_size, seq_length, -1)
        x_arousal = self.fc_arousal(x).view(batch_size, seq_length, -1)
        #reshape to original shape
        out = torch.cat((x_valence, x_arousal), dim=-1)
        return out
    
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
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-4)
