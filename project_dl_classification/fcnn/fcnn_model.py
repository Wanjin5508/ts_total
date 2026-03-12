import torch 


class FCNN(nn.Module):
    """Fully Convolutional Neural Network with global average Pooling for time-series classification."""
    def __init__(self, in_channels, nr_featuremaps, kernel_size):
        super(FCNN, self).__init__()
        
        self.nr_featuremaps = nr_featuremaps
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=nr_featuremaps, kernel_size=kernel_size, stride=1, padding='same', padding_mode='zeros', bias=False)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=nr_featuremaps, out_channels=nr_featuremaps, kernel_size=kernel_size, stride=1, padding='same', padding_mode='zeros', bias=False)
        self.relu2 = nn.ReLU()
        
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(nr_featuremaps, 2, bias=True)  
        
        self.net = nn.Sequential(
            self.conv1,
            self.relu1,
            
            self.conv2,
            self.relu2,
            
        )
        
        # Applying Kaiming Initialization to the conv layers
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu') # # preserve magnitude of the variance of the weights in the forward pass
                
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)  # Xavier Initialization for the fully connected layer

    def forward(self, x, return_conv=False):
        conv_out = self.net(x)  # Shape: (batch_size, nr_featuremaps, seq_length)
        gap = self.gap(conv_out)
        logits = self.fc1(gap.view(gap.size(0)), self.nr_featuremaps)  # Shape: (batch_size, 2)
        return logits if not return_conv else (logits, conv_out)







