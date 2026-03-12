import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.utils import weight_norm

torch.manual_seed(0)

class Chomp1d(nn.Module):
    # Module for chomping off certain number of elements from a tensor
    def __init__(self, chomp_size):
        """This is because the padding is done in the ConvLayer thus padding left and right.
        The Chomp module simply removes excess right padding from Conv1d output.
        If you left pad only before the conv, you wont need to chomp.
        
        Args:
            chomp_size (int): how many elements to chomp from right side.
        """
        
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        """Return all the elements from 1st and 2nd dimension, remove the last self.chomp_size elements from 3rd Dim
        """
        return x[:, :, :-self.chomp_size].contiguous() # contiguous for storing tensor in contigour memory block (some calculation on devices require this)
    
    
class ResidualBlock(nn.Module):
    # Building block/layer of TCN
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, drop_out=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        # nn.Conv1d applies a 1d convolution over an input signal composed of several input planes,
        # n_output defines amount of feature maps to be created --> different kernels to parametrize
        
        self.chomp1 = Chomp1d(padding)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(drop_out)
        
        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(drop_out)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.prelu1, self.dropout1,
                                 self.conv2, self.chomp2, self.prelu2, self.dropout2)
        
        self.resample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()
        
    def init_weights(self):
        # init weights with values from a gaussian distribution of mean = 0, std = 0.1
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2.weight.data.normal_(0, 0.1)
        if self.resample is not None:
            self.resample.weight.data.normal_(0, 0.1)
            
    def forward(self, x):
        out = self.net(x)
        res = x if self.resample is None else self.resample(x)
        return out + res
    
    
class TCN(nn.Module):
    def __init__(self, n_inputs=1, n_outputs=32, n_feature_maps=16, n_conv_layers=4, kernel_size=8, dilation_increase=True, drop_out=0.2):
        super(TCN, self).__init__()
        self.layers = []
        for i in range(n_conv_layers):
            dilation_factor = 2 ** i if dilation_increase else (2 ** (n_conv_layers-1-i))
            in_channels = n_inputs if i == 0 else n_feature_maps
            out_channels = n_outputs
            padding = (kernel_size - 1) * dilation_factor
            self.layers += [ResidualBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_factor, padding=padding, drop_out=drop_out)]
            
            conv_layer = weight_norm(
                nn.Conv1d(in_channels=out_channels,
                          out_channels=n_feature_maps,
                          kernel_size=1)
            )
            conv_layer.weight.data.normal_(0, 0.1)
            self.layers += [conv_layer]     # 1x1 conv to fit in to out channels
            
    def forward(self, x):
        intermediate_outputs = []       # to acess results of intermediate dilated conv layers
        for layer in self.layers:
            x = layer(x)
            intermediate_outputs += [x.clone()]
        
        return x, intermediate_outputs[1::2]
        # x contains final output of the network, intermediate_outputs[1::2] contains the n_feature_maps per conv_layer
        
        
class TCNEncoder(nn.Module):
    def __init__(self, n_inputs, tcn_outputs=32, n_feature_maps=16, n_conv_layers=4, kernel_size=8, dilation_increase=True, drop_out=0.2, enc_dim=4, enc_pool_size=32):
        super(TCNEncoder, self).__init__()
        self.tcn = TCN(n_inputs, tcn_outputs, n_feature_maps, n_conv_layers, kernel_size, dilation_increase, drop_out)
        self.conv = weight_norm(
            nn.Conv1d(in_channels=n_conv_layers*n_feature_maps, out_channels=enc_dim, kernel_size=1)
        )   
        self.avgpooling = nn.AvgPool1d(kernel_size=enc_pool_size, stride=enc_pool_size)
        self.init_weights()
        
    def forward(self, x):
        x, outs_dil_conv_layers = self.tcn(x)
        outs_dil_conv_layers = torch.cat(outs_dil_conv_layers, dim=1)   # stack outputs as tensor with n_conv_layers channels
        conv = self.conv(outs_dil_conv_layers)
        out = self.avgpooling(conv)
        return outs_dil_conv_layers, out
    
    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.1)
        
class TCNDecoder(nn.Module):
    def __init__(self, n_inputs=4, tcn_outputs=32, n_feature_maps=16, n_conv_layers=4, kernel_size=8, dilation_increase=False, drop_out=0.2, dec_dim=1, enc_pool_size=32):
        super(TCNDecoder, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=enc_pool_size, mode='nearest')
        self.tcn = TCN(n_inputs, tcn_outputs, n_feature_maps, n_conv_layers, kernel_size, dilation_increase, drop_out)
        self.conv = weight_norm(
            nn.Conv1d(in_channels=n_conv_layers*n_feature_maps, out_channels=dec_dim, kernel_size=1)
            
        )
        
        self.activation = nn.PReLU()    # TODO change to SRelU to limit output range
        self.init_weight()
        
    def forward(self, x):
        upsampled_x = self.upsampling(x)
        _, outs_dil_conv_layers = self.tcn(upsampled_x)
        outs_dil_conv_layers = torch.cat(outs_dil_conv_layers, dim=1)
        decoded = self.activation(self.conv(outs_dil_conv_layers))
        return outs_dil_conv_layers, decoded
    
    def init_weight(self):
        self.conv.weight.data.normal_(0, .1)
        
class TCNAE(nn.Module):
    def __init__(self, n_inputs=1, tcn_outputs=32, n_feature_maps=16, n_conv_layers=4, kernel_size=8, drop_out=0.2, enc_dim=4, enc_pool_size=32):
        super(TCNAE, self).__init__()
        
        self.encoder = TCNEncoder(n_inputs=n_inputs, tcn_outputs=tcn_outputs, n_feature_maps=n_feature_maps, n_conv_layers=n_conv_layers, kernel_size=kernel_size, dilation_increase=True, drop_out=drop_out, enc_dim=enc_dim, enc_pool_size=enc_pool_size)
        self.decoder = TCNDecoder(n_inputs=enc_dim, tcn_outputs=tcn_outputs, n_feature_maps=n_feature_maps, n_conv_layers=n_conv_layers, kernel_size=kernel_size, dilation_increase=False, drop_out=drop_out, dec_dim=n_inputs, enc_pool_size=enc_pool_size)
        
    def forward(self, x):
        self.res_enc_dil_conv_layers, self.encoded = self.encoder(x)
        self.res_dec_dil_conv_layers, self.decoded = self.decoder(self.encoded)
        return self.decoded
        
        
if __name__ == "__main__":
    input_data = torch.rand(1, 1, 128)






