import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
# Initialize VGGish

# Freeze all layers in VGGish to prevent them from being trained


class EmotionRegressor(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.vggish_model =  torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.vggish_model.eval()
        for param in self.vggish_model.parameters():
            param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.vggish_model(x)
        x = self.flatten(x, start_dim=1)
        x = self.regressor(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(train_dataset, batch_size=32)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(val_dataset, batch_size=32)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(test_dataset, batch_size=32)

