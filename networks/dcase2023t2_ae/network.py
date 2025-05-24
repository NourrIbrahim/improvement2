import torch
from torch import nn

class AENet(nn.Module):
    def __init__(self, input_dim, block_size):
        super(AENet, self).__init__()
        self.input_dim = input_dim

        # Covariance matrices (only needed for Mahalanobis; retained for compatibility)
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.BatchNorm1d(8, momentum=0.01, eps=1e-03),
            nn.ReLU()
        )

        # Projector module (added for better latent representation)
        self.projector = nn.Sequential(
            nn.Linear(8, 8),
            nn.BatchNorm1d(8, momentum=0.01, eps=1e-03),
            nn.ReLU()
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        z = self.encoder(x)
        z_proj = self.projector(z)  # project latent features
        x_hat = self.decoder(z_proj)
        return x_hat, z_proj  # you may also return z if you want to apply SMOTE on it
