import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, weights=None):
        super(MLP, self).__init__()
        layers = []
        for in_size, out_size in zip([input_size] + hidden_sizes[:-1],
                                     hidden_sizes):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        # 最后一层不加激活函数
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

        # 权重层
        self.weights = weights

    def forward(self, x):
        if self.weights is not None:
            x_weights = x * self.weights
        else:
            x_weights = x
        return self.model(x_weights)
