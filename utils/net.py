from torch import nn

from .common import weight_init


MLP_NORM = {"BN": nn.BatchNorm1d, "LN": nn.LayerNorm, "None": None}


class MLP(nn.Module):
    def __init__(self, dims, activation=nn.ReLU, last_activation=False, norm="None", last_norm=False):
        super().__init__()
        norm = MLP_NORM[norm]
        self.output_dim = dims[-1]
        layers = []
        for k in range(len(dims) - 1):
            layers.append(nn.Linear(dims[k], dims[k + 1]))
            if norm is not None:
                layers.append(norm(dims[k + 1]))
            layers.append(activation())
        if not last_activation:
            layers.pop()
        if (norm is not None) and (not last_norm):
            layers.pop()
        self.mlp = nn.Sequential(*layers)
        self.apply(weight_init)

    def forward(self, obs):
        return self.mlp(obs.flatten(1))
