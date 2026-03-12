
def conv1d_out_len(L, k, s=1, p=0, d=1):
    return ((L + 2*p - d*(k-1)) // s) + 1


class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, num_classes, hidden_size=64, num_layers=2, bidirectional=True, dropout=0.2, agg='attn', input_size=1):
        super().__init__()

        self.fcnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, stride=2, padding=2, dilation=1),
            # nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, dilation=1),
            # nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1),
            # nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # removed avg pooling, to avoid errors from padding in dataset
        # self.gap = nn.AdaptiveAvgPool1d(10)

        self.lstm = nn.LSTM(
            input_size=128,  # channel size from the last conv1d
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.hidden_out = hidden_size * (2 if bidirectional else 1)
        self.agg = agg

        self.head = nn.Sequential(
            # nn.LayerNorm(self.hidden_out),
            nn.Linear(self.hidden_out, self.hidden_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_out // 2, num_classes)
        )

        self._conv_cfg = [
            dict(k=5, s=2, p=2, d=1),
            dict(k=5, s=2, p=2, d=1),
            dict(k=3, s=2, p=1, d=1),
        ]

    @torch.no_grad()
    def _lengths_after_conv_stack(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        lengths: (N,), the original lengths, before padding
        return: the real lengths after conv stack
        """
        L = lengths.clone()
        for cfg in self._conv_cfg:
            L = conv1d_out_len(L, cfg['k'], cfg['s'], cfg['p'], cfg['d'])
            L = L.clamp_min(1)
        return L

    def forward(self, x, lengths):
        """
        cnn input x shape: (N, C_in=1, L_max)
        lengths: (N, )
        """
        # 1. CNN stack, the last dim is time points
        x = self.fcnn(x)  # -> (N, 128, L3_actual_before_T)
        L3_actual = x.size(-1)

        # x = x.transpose(1,2).contiguous() # (N, L3, 128)

        # TODO better except handler
        # print(f"x size after fcnn: {x.size()}")
        # assert x.size(1) == 128, f"Conv out channels = {x.size()} != 128"

        # 2. get the effective lengths after conv
        lengths_prime = self._lengths_after_conv_stack(lengths)  # (N, )
        # print(f"lengths_prime = {lengths_prime}")
        lengths_prime = torch.minimum(lengths_prime, torch.as_tensor(L3_actual, device=lengths_prime.device))
        lengths_prime = lengths_prime.clamp_min(1)

        # 3.
        # lstm takes input of shape (batch_size, seq_len, input_size) <- (N, L3_actual, 128)
        x = x.transpose(1, 2).contiguous()

        # 4. pack: only pass the effective time steps to lstm
        packed = pack_padded_sequence(x, lengths_prime.to("cpu"), batch_first=True, enforce_sorted=False)

        packed_out, (h_n, c_n) = self.lstm(packed)

        # get the original order in a batch
        # inv = packed.unsorted_indices.to(h_n.device)  # (N,)

        if self.lstm.bidirectional:
            # get forward and backward from the last layer, 内部排序过的
            h_fwd = h_n[-2]  # (N, hidden_size)
            h_bwd = h_n[-1]  # (N, hidden_size)
            feat = torch.cat([h_fwd, h_bwd], dim=1)
        else:
            feat = h_n[-1]

        # out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=L3_actual)  # (N, L3_actual, hidden_out)

        # # 5. get the last effective steps of every sample
        # idx = (lengths_prime - 1).clamp_min(0).to(out.device)  # (N,)
        # idx = idx.view(-1, 1, 1).expand(-1, 1, out.size(2))  # (N, 1, H)
        # feat = out.gather(dim=1, index=idx).squeeze(1)  # (N, H), H=hidden_out for bidirectional, else hidden_size

        # 6. classifier head
        out_logits = self.head(feat)  # (N, num_classes)

        return out_logits










