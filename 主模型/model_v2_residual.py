"""
主模型 v2 + 残差：CNN 多尺度特征跨过 CrossAttention 直连 FC

架构:
  输入 [B, 3, N]
    │
    └──→ Conv1d(3→hidden, k=1)  ← 1×1 升维
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
  CNN3   CNN5   BiGRU
  k=3    k=5    双向
    │      │      │
  SE      SE     │
    │      │      │
    └─Concat─────┘
         │   │    │
      [2·h]  │   [h]
         │   │    │
         │   K,V  Q
         │   └─CrossAttn──┘
         │         │
         │      [B,hidden]
         │         │
         └──→ (+) ──→ FC → SOH
           (残差连接)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════
# SE 通道注意力模块
# ═══════════════════════════════════════════════
class SEBlock(nn.Module):
    """Squeeze-and-Excitation：自动学习每个通道的重要性权重。"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, L]
        b, c, _ = x.shape
        w = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, C]
        w = self.fc(w).unsqueeze(-1)                  # [B, C, 1]
        return x * w


# ═══════════════════════════════════════════════
# CNN 分支
# ═══════════════════════════════════════════════
class CNNBranch(nn.Module):
    """1D CNN 分支：多层卷积 + BN + ReLU + SE + 全局池化，层数可配"""

    def __init__(self, hidden_dim=64, kernel_size=3, cnn_layers=3, se_reduction=4):
        super().__init__()
        padding = kernel_size // 2

        layers = []
        for i in range(cnn_layers):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

        self.se = SEBlock(hidden_dim, reduction=se_reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, hidden, N]
        x = self.conv(x)              # [B, hidden, N]
        x = self.se(x)                # [B, hidden, N]  通道加权
        x = self.pool(x).squeeze(-1)  # [B, hidden]
        return x


# ═══════════════════════════════════════════════
# BiGRU 分支
# ═══════════════════════════════════════════════
class BiGRUBranch(nn.Module):
    """双向 GRU 分支：BiGRU → 投影 → 全局池化"""

    def __init__(self, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        # x: [B, hidden, N] → [B, N, hidden]
        x = x.transpose(1, 2)
        out, _ = self.gru(x)                       # [B, N, 2*hidden]
        out = self.proj(out)                       # [B, N, hidden]
        out = out.transpose(1, 2)                  # [B, hidden, N]
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)  # [B, hidden]
        return out


# ═══════════════════════════════════════════════
# GRU → CNN 交叉注意力 (时序引导空间)
# ═══════════════════════════════════════════════
class TemporalSpatialCrossAttention(nn.Module):
    """
    时序引导空间注意力：
    Q = BiGRU (宏观退化趋势)
    K = V = CNN concat (多尺度 IC 峰形态)

    GRU 用它的"全局视角"去查询 CNN 的"局部证据"——
    在物理上等价于：利用电池老化趋势自动定位 IC 峰中最关键的形态变化。
    """

    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # K/V 是 CNN concat，维度为 2*hidden；Q 是 GRU，维度为 hidden
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,        # Q 的维度
            kdim=hidden_dim * 2,         # K 的维度
            vdim=hidden_dim * 2,         # V 的维度
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, gru_feat, cnn_concat):
        """
        gru_feat:   [B, hidden]        — Q，宏观退化趋势
        cnn_concat: [B, 2*hidden]      — K, V，多尺度 IC 峰形态
        返回:       [B, hidden]        — 经过注意力增强的 GRU 特征
        """
        q = gru_feat.unsqueeze(1)       # [B, 1, hidden]
        kv = cnn_concat.unsqueeze(1)    # [B, 1, 2*hidden]

        attn_out, _ = self.attn(q, kv, kv)  # [B, 1, hidden]
        attn_out = attn_out.squeeze(1)      # [B, hidden]

        # 残差 + LayerNorm
        out = self.norm(gru_feat + attn_out)
        return out


# ═══════════════════════════════════════════════
# 主模型
# ═══════════════════════════════════════════════
class MainSOHModelV2Residual(nn.Module):
    """
    主模型 v2 + 残差：CNN 多尺度特征绕过 CrossAttention 直连 FC

    参数:
        in_channels:   输入特征维度，默认 3
        seq_len:       时间步长，默认 32
        hidden_dim:    各分支隐藏维度，默认 64
        num_heads:     交叉注意力头数，默认 4
        gru_layers:    BiGRU 层数，默认 1
        se_reduction:  SE 模块降维比，默认 4
        dropout:       FC 层 Dropout，默认 0.2
        output_dim:    输出维度，默认 1 (SOH)
    """

    def __init__(
        self,
        in_channels=3,
        seq_len=32,
        hidden_dim=64,
        num_heads=4,
        cnn_layers=3,
        gru_layers=1,
        gru_dropout=0.1,
        se_reduction=4,
        dropout=0.2,
        output_dim=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # ── 1×1 升维层 ──
        self.lift = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)

        # ── 三个并行分支 ──
        self.cnn3 = CNNBranch(hidden_dim, kernel_size=3, cnn_layers=cnn_layers, se_reduction=se_reduction)
        self.cnn5 = CNNBranch(hidden_dim, kernel_size=5, cnn_layers=cnn_layers, se_reduction=se_reduction)
        self.bigru = BiGRUBranch(hidden_dim, num_layers=gru_layers, dropout=gru_dropout)

        # ── 时序引导空间注意力 ──
        self.cross_attn = TemporalSpatialCrossAttention(hidden_dim, num_heads)

        # ── CNN 残差投影：[2*hidden] → [hidden] ──
        self.cnn_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── 回归头 ──
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
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        输入: x [B, in_channels, seq_len]  = [B, 3, N]
        输出: out [B, output_dim]           = [B, 1]
        """
        # 1×1 升维
        x = self.lift(x)  # [B, hidden, N]

        # 三分支并行
        f3 = self.cnn3(x)   # [B, hidden]
        f5 = self.cnn5(x)   # [B, hidden]
        fg = self.bigru(x)  # [B, hidden]

        # CNN 双路拼接 → K/V + 残差支路
        cnn_concat = torch.cat([f3, f5], dim=-1)  # [B, 2*hidden]
        cnn_residual = self.cnn_proj(cnn_concat)   # [B, hidden]

        # 时序引导空间注意力：Q=GRU, K/V=CNN_concat
        attn_out = self.cross_attn(fg, cnn_concat)  # [B, hidden]

        # 残差融合：注意力 + CNN 直连
        out = self.fc(attn_out + cnn_residual)  # [B, 1]
        return out


# ═══════════════════════════════════════════════
# 快速测试
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = MainSOHModelV2Residual(
        in_channels=3,
        seq_len=32,
        hidden_dim=64,
        num_heads=4,
        gru_layers=1,
        gru_dropout=0.1,
        se_reduction=4,
        dropout=0.2,
    ).to(device)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total:,}")

    x = torch.randn(8, 3, 32).to(device)
    y = model(x)
    print(f"输入: {x.shape} → 输出: {y.shape}")

    loss = y.mean()
    loss.backward()
    print("反向传播: OK")
