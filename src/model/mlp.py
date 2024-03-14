import torch.nn as nn
from src.model.weight_init import trunc_normal_

ACT_FN = {
    'RELU': nn.ReLU(),
    'ELU': nn.ELU(),
    'LEAKY_RELU': nn.LeakyReLU()
}


class MLP(nn.Module):
    def __init__(self, mlp_cfg, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = mlp_cfg.INPUT_SIZE

        if mlp_cfg.LAYERS > 0:
            layers = []
            layers.append(nn.Linear(input_size, mlp_cfg.HIDDEN_SIZE))
            layers.append(nn.Dropout(mlp_cfg.DROPOUT))
            layers.append(ACT_FN[mlp_cfg.ACT_FN])
            for _ in range(mlp_cfg.LAYERS - 1):
                layers.append(nn.Linear(mlp_cfg.HIDDEN_SIZE, mlp_cfg.HIDDEN_SIZE))
                layers.append(nn.Dropout(mlp_cfg.DROPOUT))
                layers.append(ACT_FN[mlp_cfg.ACT_FN])
            self.feature = nn.Sequential(*layers)
            self.output_layer = nn.Linear(mlp_cfg.HIDDEN_SIZE, mlp_cfg.OUTPUT_SIZE)
        else:
            self.feature = None
            self.output_layer = nn.Linear(input_size, mlp_cfg.OUTPUT_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def model_params(self):
        return list(self.parameters())

    def forward(self, x, return_feature=False):

        if self.feature is not None:
            x = self.feature(x)
        out = self.output_layer(x)

        if return_feature:
            return out, x
        else:
            return out