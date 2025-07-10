import torch
from torch import nn
from utils.almt_layer import Transformer


class Reconstruct(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.visual_reconstructor=nn.Sequential(
            nn.Linear(1024,2048),
            # nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        )
        self.audio_reconstructor=nn.Sequential(
            nn.Linear(1024,2048),
            # nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        )
        self.criterion_v = nn.MSELoss()
        self.criterion_a = nn.MSELoss()

        
    def forward(self,v_feature,v_raw_feature,a_feature,a_raw_feature):
        v_recon_feature=self.visual_reconstructor(v_feature)
        a_recon_feature=self.audio_reconstructor(a_feature)
        recon_v_loss=self.criterion_v(v_recon_feature,v_raw_feature)
        recon_a_loss=self.criterion_a(a_recon_feature,a_raw_feature)
        return recon_v_loss,recon_a_loss




