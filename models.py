import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, 
                num_layers=3,
                width=[32, 64, 128],
                kernel_size=3,
                activation=nn.ReLU,
                pooling=nn.MaxPool2d, 
                dropout=0.0, 
                norm_layer=None, 
                in_channels=3,
                n_output=100):
        super().__init__()
        assert num_layers <= 4, "Number of Layers must be less than 5."
        assert len(width) >= num_layers, "Width list must match or exceed num_layers."
        self.num_layers = num_layers
        self.activation = activation()
        self.pool = pooling(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            out_channels = width[i]
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            self.norms.append(norm_layer(out_channels) if norm_layer else nn.Identity())
            in_channels = out_channels  # Update input channels for next layer

        fc_input_dim = width[num_layers - 1] * (32 // (2 ** num_layers)) ** 2 # max: 4 layers
        
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, n_output)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.convs[i](x)
            x = self.norms[i](x)
            x = self.activation(x)
            x = self.pool(x) 

        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
