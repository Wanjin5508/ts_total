
from typing import Union  

import torch
from torch import nn
import torch.nn.functional as F

from config import DOWN_SAMPLING_KERNELS


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(DoubleConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=9, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=9, padding="same" ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer(x)
    
class DownSampleBlock(nn.Module):
    def __init__(self, kernel_size):
        super(DownSampleBlock, self).__init__()
        self.layer = nn.MaxPool1d(kernel_size=kernel_size)
        
    def forward(self, x):
        return self.layer(x)
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(UpSampleBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=kernel_size, stride=kernel_size),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, dilation=9, padding="same" ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU( )
        )       
        
    def forward(self, x_enc, x_dec):
        up = self.layer(x_dec)
        concat = crop_concat(x_enc, up)
        return concat
    
def crop_concat(feature_map_encoder, feature_map_decoder):
    croped = F.interpolate(feature_map_encoder, feature_map_decoder.size(-1))
    assert croped.size() == feature_map_decoder.size(), "Sizes do not match for concatenation"
    return torch.cat([feature_map_decoder,  croped], dim=1)


class OneDimConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OneDimConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, num_classes, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.layer(x)
    
    
class UTime(nn.Module):
    def __init__(self, num_classes, depth=4, in_channels=1, base_channels=16, seg_num=-1):
        super(UTime, self).__init__()
        self.depth = depth
        assert seg_num != -1, "seg_num in UTime must be positive!"
        self.seg_num = seg_num

        self.encoder_list = nn.ModuleList()
        self.down_sampling_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        self.up_sampling_list = nn.ModuleList()
        
        out_ch = 0
        
        # encoders, down side
        prev_channels = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.encoder_list.append(
                DoubleConvBlock(in_channels=prev_channels, out_channels=out_ch)
            )
            self.down_sampling_list.append(
                DownSampleBlock(DOWN_SAMPLING_KERNELS[i])
            )
            
            prev_channels = out_ch
            
        # bottom
        self.bottom = DoubleConvBlock(prev_channels, prev_channels*2)
        
        # decoders, up side
        for i in reversed(range(depth)):
            out_ch = base_channels * (2**i)
            in_ch = out_ch * 2
            
            self.up_sampling_list.append(
                UpSampleBlock(in_channels=in_ch, out_channels=out_ch, kernel_size=DOWN_SAMPLING_KERNELS[i])
            )
            
            self.decoder_list.append(
                DoubleConvBlock(in_channels=in_ch, out_channels=out_ch)
            )
            
        # between last decoder and classfier
        self.one_by_one_conv = OneDimConv(out_ch, num_classes)
        
        # classifier
        self.one_by_one_conv_over_segments = nn.Conv1d(in_channels=num_classes, out_channels=num_classes, kernel_size=1, padding='same')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        input_length = x.size(-1)
        encoder_features = []
        
        # encoder side
        for i in range(self.depth):
            x = self.encoder_list[i](x)
            encoder_features.append(x)
            x = self.down_sampling_list[i](x)
            
        # bottom
        x = self.bottom(x)
        
        # decoder side
        for i in range(self.depth):
            x = self.up_sampling_list[i](x_enc=encoder_features[-(i+1)], x_dec=x)
            x = self.decoder_list[i](x)
            
        one_by_one_conv_out = self.one_by_one_conv(x)
        padding_value = (input_length - one_by_one_conv_out.size(-1)) // 2
        self.padding = nn.ZeroPad1d((padding_value, padding_value))
        padded_scores = self.padding(one_by_one_conv_out)
        
        # reshape to [T, i, K]
        chunk_in_segments = torch.reshape(
            padded_scores,
            (padded_scores.size(0), padded_scores.size(1), self.seg_num, -1)
        )
        
        print(f"reshape back to seg nums: {chunk_in_segments.size()}")

        # average pooling -> [T, K]
        batch_size = chunk_in_segments.size(0)
        num_classes = chunk_in_segments.size(1)
        num_segments = chunk_in_segments.size(2)
        
        self.avg_pooling = nn.AvgPool1d(kernel_size=chunk_in_segments.size(-1)) # ? 序列长度 是 -1？
        avg_pooling_output = self.avg_pooling(
            torch.reshape(
                chunk_in_segments,
                (batch_size*num_classes*num_segments, -1)
        ))
        
        avg_pooling_output = torch.reshape(avg_pooling_output, (batch_size, num_classes, num_segments))
        print(f"avg_pooling_output: {avg_pooling_output.size()}")

        logits = self.one_by_one_conv_over_segments(avg_pooling_output)
        
        probs = self.softmax(logits)
        return logits, probs

if __name__ == "__main__":
    model = UTime(2, 4, seg_num=1000)
    x = torch.ones((1, 1, 105000), dtype=torch.float32)
    out = model(x)

