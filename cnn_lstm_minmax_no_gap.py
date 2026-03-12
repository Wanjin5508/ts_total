import torch
import torch.nn as nn

class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, num_classes, hidden_size=64, num_layers=2, bidirectional=True, dropout=0.2, input_size=1):
        super(CNN_LSTM_Classifier, self).__init__()
        
        self.fcnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, padding=1),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=1),
            nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool1d(output_size=10)
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, 
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        self.hidden_size = hidden_size
        self.bidrectional = bidirectional
        self.hidden_out = hidden_size * 2 if bidirectional else hidden_size
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        x = self.fcnn(x)
        x = self.gap(x)
        x = x.permute(0, 2, 1)  # Change shape to (batch, seq_len, features) for LSTM
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        lstm_out = h_n[-1] # if not self.bidrectional else torch.cat((h_n[-2], h_n[-1]), dim=1)
        
        out = self.head(lstm_out)
        return out

################################## 以下为正确代码 #####################################

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ---- 可选：通用的长度推导函数（便于以后你改卷积超参） ----
def conv1d_out_len(L, k, s=1, p=0, d=1):
    # L, k, s, p, d 都是标量或张量（L 为 LongTensor）
    return ((L + 2*p - d*(k-1) - 1) // s) + 1

class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, num_classes, hidden_size=64, num_layers=2, bidirectional=True, dropout=0.2, input_size=1):
        super(CNN_LSTM_Classifier, self).__init__()

        # 3 个 1D 卷积（与你原代码一致：k=5, p=1, s=1）
        self.fcnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, padding=2, stride=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, dilation=1),
            nn.ReLU()
        )

        # ❌ 移除时间向 AdaptiveAvgPool1d，避免 padding 污染
        # self.gap = nn.AdaptiveAvgPool1d(output_size=10)

        self.lstm = nn.LSTM(
            input_size=128,                # 来自最后一层 Conv1d 的通道数
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # 官方约定：单层 LSTM 的 dropout 不生效
            batch_first=True
        )

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.hidden_out = hidden_size * (2 if bidirectional else 1)

        # ⚠️ LayerNorm 和 Linear 的输入维度应为 hidden_out（不是 hidden_size）
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_out),
            nn.Linear(self.hidden_out, self.hidden_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_out // 2, num_classes)
        )

        # 记录卷积配置（用于推导卷积后的有效长度）
        self._conv_cfg = [
            dict(k=5, s=2, p=2, d=1),
            dict(k=5, s=2, p=2, d=1),
            dict(k=5, s=2, p=1, d=1),
        ]

    @torch.no_grad()
    def _lengths_after_conv_stack(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        lengths: (N,) 原始有效长度（padding 前）
        返回：卷积堆叠后的有效长度 (N,)
        """
        L = lengths.clone()
        for cfg in self._conv_cfg:
            L = conv1d_out_len(L, cfg['k'], cfg['s'], cfg['p'], cfg['d'])
            L = L.clamp_min(1)  # 防止出现 0/负数
        return L

    def forward(self, x, lengths):
        """
        x: (N, C_in=1, L_max)  —— 按批次 pad 到 L_max
        lengths: (N,)          —— 每个样本 padding 前的有效长度
        """
        # 1) CNN 堆叠（时间维仍在最后一维）
        x = self.fcnn(x)              # (N, 128, L3)

        L3_actual = x.size(-1)            
        
        # 2) 计算卷积后的有效步长
        lengths_prime = self._lengths_after_conv_stack(lengths)  # (N,)
        lengths_prime = torch.minimum(lengths_prime, torch.tensor(L3_actual, device=lengths_prime.device))

        # 3) 转给 LSTM: (N, L3, 128)
        x = x.transpose(1, 2).contiguous()

        # 4) pack：只把有效步交给 LSTM
        # 注意：pack 的 lengths 最稳妥用 CPU 张量
        packed = pack_padded_sequence(x, lengths_prime.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        # out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (N, L3, hidden_out)

        # 5) 读出：取每个样本的最后有效步（也可改为掩码平均，见下面注释）
        # idx = (lengths_prime - 1).clamp_min(0).to(out.device)       # (N,)
        # feat = out[torch.arange(out.size(0), device=out.device), idx]  # (N, hidden_out)

        # 另一种更鲁棒的读出：掩码时间平均（可替换上面三行）
        # L3 = out.size(1)
        # mask = (torch.arange(L3, device=out.device)[None, :] < lengths_prime.to(out.device)[:, None]).float()
        # feat = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        
        if self.lstm.bidirectional:
            feat = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (N, hidden_out)
        else:
            feat = h_n[-1]  # (N, hidden_out)

        # 6) 分类头（含 LayerNorm，仅归一化最后一维）
        out_logits = self.head(feat)  # (N, num_classes)
        return out_logits


import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    batch: List[(x, y)]
      x: Tensor (C, L_i)  —— 单样本；你已有的 min-max 等预处理应在 __getitem__ 里完成
      y: 标量/张量（标签）
    返回:
      xs: (N, C, L_max), ys: (N,), lengths: (N,)
    """
    xs, ys, lengths = [], [], []
    for x, y in batch:
        assert x.ndim == 2 and x.shape[0] >= 1, "expect (C, L)"
        C, Li = x.shape
        xs.append(x.transpose(0, 1))  # (L_i, C)
        ys.append(torch.as_tensor(y, dtype=torch.long))
        lengths.append(Li)

    xs = pad_sequence(xs, batch_first=True, padding_value=0.0)  # (N, L_max, C)
    xs = xs.transpose(1, 2).contiguous()                        # -> (N, C, L_max)
    ys = torch.stack(ys, dim=0)                                 # (N,)
    lengths = torch.as_tensor(lengths, dtype=torch.long)        # (N,)
    return xs, ys, lengths

model = CNN_LSTM_Classifier(num_classes=2, hidden_size=64, num_layers=2, bidirectional=True, dropout=0.2, input_size=1)
model.to(device)

for xs, ys, lengths in train_loader:
    xs, ys, lengths = xs.to(device), ys.to(device), lengths.to(device)
    logits = model(xs, lengths)            # (N, 2)
    loss = nn.CrossEntropyLoss()(logits, ys)
    ...


def train_epoch(model, loader, optimizer, criterion, device='cuda'):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for xs, ys, lengths in loader:
        xs, ys, lengths = xs.to(device), ys.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(xs, lengths)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xs.size(0)
        total += xs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == ys).sum().item()
    return total_loss / total, correct / total