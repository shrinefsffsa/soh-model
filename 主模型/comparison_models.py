"""
对比模型：纯 BiGRU / 纯 Multi-Scale TCN / CNN∥BiGRU concat（无 CrossAttn）

每个模型接口与 MainSOHModelV2 一致，可直接替换 bayesian_search_v2.py 的 import。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════
# 共享组件
# ═══════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


# ═══════════════════════════════════════════════
# 1. 纯 BiGRU
# ═══════════════════════════════════════════════
class PureBiGRUModel(nn.Module):
    """
    纯双向 GRU：
      输入 [B, 3, N] → 1×1 升维 → BiGRU → 全局池化 → FC → SOH
    """
    def __init__(self, in_channels=3, seq_len=32, hidden_dim=64,
                 num_heads=4, cnn_layers=3, gru_layers=1,
                 gru_dropout=0.1, se_reduction=4, dropout=0.2, output_dim=1):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)

        self.gru = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=gru_layers, batch_first=True, bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, 3, N]
        x = self.lift(x)                           # [B, hidden, N]
        x = x.transpose(1, 2)                      # [B, N, hidden]
        out, _ = self.gru(x)                       # [B, N, 2*hidden]
        out = self.proj(out)                       # [B, N, hidden]
        out = out.transpose(1, 2)                  # [B, hidden, N]
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)  # [B, hidden]
        return self.fc(out)


# ═══════════════════════════════════════════════
# 2. 纯 Multi-Scale TCN
# ═══════════════════════════════════════════════
class TCNBlock(nn.Module):
    """单层 TCN：膨胀卷积 + BN + ReLU"""
    def __init__(self, hidden_dim, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                              padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class PureTCNModel(nn.Module):
    """
    纯多尺度 TCN：
      输入 [B, 3, N] → 1×1 升维 → TCN3∥TCN5∥TCN7 → Concat → 全局池化 → FC → SOH
    """
    def __init__(self, in_channels=3, seq_len=32, hidden_dim=64,
                 num_heads=4, cnn_layers=3, gru_layers=1,
                 gru_dropout=0.1, se_reduction=4, dropout=0.2, output_dim=1):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)

        # 三路 TCN，不同 kernel
        for ks in [3, 5, 7]:
            layers = []
            for i in range(cnn_layers):
                layers.append(TCNBlock(hidden_dim, kernel_size=ks, dilation=2**i))
            setattr(self, f"tcn{ks}", nn.Sequential(*layers))
            setattr(self, f"se{ks}", SEBlock(hidden_dim, reduction=se_reduction))

        self.proj = nn.Linear(hidden_dim * 3, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.lift(x)  # [B, hidden, N]

        # 三路 TCN 并行
        feats = []
        for ks in [3, 5, 7]:
            f = getattr(self, f"tcn{ks}")(x)
            f = getattr(self, f"se{ks}")(f)
            f = F.adaptive_avg_pool1d(f, 1).squeeze(-1)  # [B, hidden]
            feats.append(f)

        concat = torch.cat(feats, dim=-1)  # [B, 3*hidden]
        out = F.relu(self.proj(concat))   # [B, hidden]
        return self.fc(out)


# ═══════════════════════════════════════════════
# 3. CNN∥BiGRU Concat（无 CrossAttn / 无注意力）
# ═══════════════════════════════════════════════
class CNNBranch(nn.Module):
    """同 model_v2 的 CNN 分支"""
    def __init__(self, hidden_dim, kernel_size, cnn_layers, se_reduction):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        for _ in range(cnn_layers):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(hidden_dim, reduction=se_reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return self.pool(x).squeeze(-1)


class BiGRUBranch(nn.Module):
    """同 model_v2 的 BiGRU 分支"""
    def __init__(self, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        out = self.proj(out)
        out = out.transpose(1, 2)
        return F.adaptive_avg_pool1d(out, 1).squeeze(-1)


class NoAttentionModel(nn.Module):
    """
    CNN∥BiGRU 简单拼接（无 CrossAttn）：
      跟你的模型完全一样的编码器，但去掉 CrossAttention 直接 Concat → FC
    """
    def __init__(self, in_channels=3, seq_len=32, hidden_dim=64,
                 num_heads=4, cnn_layers=3, gru_layers=1,
                 gru_dropout=0.1, se_reduction=4, dropout=0.2, output_dim=1):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.cnn3 = CNNBranch(hidden_dim, 3, cnn_layers, se_reduction)
        self.cnn5 = CNNBranch(hidden_dim, 5, cnn_layers, se_reduction)
        self.bigru = BiGRUBranch(hidden_dim, gru_layers, gru_dropout)

        # CNN*2 + GRU → 简单拼接
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.lift(x)
        f3 = self.cnn3(x)     # [B, hidden]
        f5 = self.cnn5(x)     # [B, hidden]
        fg = self.bigru(x)    # [B, hidden]
        concat = torch.cat([f3, f5, fg], dim=-1)  # [B, 3*hidden]
        return self.fc(concat)


# ═══════════════════════════════════════════════
# 快速测试
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for Model in [PureBiGRUModel, PureTCNModel, NoAttentionModel]:
        model = Model(in_channels=3, seq_len=32, hidden_dim=64).to(device)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x = torch.randn(8, 3, 32).to(device)
        y = model(x)
        print(f"{Model.__name__:25s}  参数量: {n:>10,}  输入: {list(x.shape)}  输出: {list(y.shape)}")
        y.mean().backward()
