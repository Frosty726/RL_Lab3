from torch import nn

class DQN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, out_feats)
        )

    def forward(self, x):
        # return logits
        return self.net(x)