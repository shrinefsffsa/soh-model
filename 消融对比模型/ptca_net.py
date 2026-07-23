"""
PTCA-Net: Parallel Temporal Convolutional Attention Network

基于 Cha et al., "Parallel temporal convolutional attention with multi-scale
feature fusion for high-accuracy lithium-ion battery SOH estimation from
fragmented charging segments", Journal of Energy Storage 141 (2026) 119430.

架构:
  输入 [B, 3, N]
    │
    1D Conv 升维 → [B, H, N]
    │
    ├─ TCN k=3 (膨胀因果卷积 + GeLU + 残差) ─┐
    ├─ TCN k=5 ────────────────────────────────┼─ Concat → [B, 3*H, N]
    └─ TCN k=7 ────────────────────────────────┘
    │
    Transformer Encoder (Self-Attn MHA + FFN + LayerNorm + 残差)
    │
    全局平均池化 → FC → SOH

接口与 MainSOHModelV2 兼容，可直接替换 bayesian_search_v2.py 的 import。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════
# TCN 时序块（膨胀因果卷积）
# ═══════════════════════════════════════════════
class TemporalBlock(nn.Module):
    """单层 TCN：膨胀因果卷积 + GeLU + 残差连接"""

    def __init__(self, hidden_dim, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # 因果卷积 padding
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # 1×1 卷积做残差对齐（维度相同，但为了规范还是加一个）
        self.residual = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        # x: [B, C, L]
        out = self.conv(x)
        out = out[:, :, :x.shape[-1]]  # 裁剪 padding，保持因果性
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.bn(out)

        res = self.residual(x)
        return out + res  # 残差


# ═══════════════════════════════════════════════
# 并行 TCN 模块（三路多尺度）
# ═══════════════════════════════════════════════
class ParallelTCN(nn.Module):
    """
    三路并行 TCN，kernel size 分别为 3, 5, 7。
    每路 stack cnn_layers 个 TemporalBlock，膨胀因子逐层 ×2。
    """

    def __init__(self, hidden_dim=64, cnn_layers=3, tcn_dropout=0.1):
        super().__init__()

        self.kernel_sizes = [3, 5, 7]
        self.branches = nn.ModuleList()

        for ks in self.kernel_sizes:
            blocks = []
            for i in range(cnn_layers):
                blocks.append(TemporalBlock(
                    hidden_dim,
                    kernel_size=ks,
                    dilation=2 ** i,
                    dropout=tcn_dropout,
                ))
            self.branches.append(nn.Sequential(*blocks))

    def forward(self, x):
        # x: [B, H, N]
        feats = [branch(x) for branch in self.branches]  # 各 [B, H, N]
        return torch.cat(feats, dim=1)  # [B, 3*H, N]


# ═══════════════════════════════════════════════
# Transformer Encoder
# ═══════════════════════════════════════════════
class TransformerEncoderBlock(nn.Module):
    """单层 Transformer Encoder：MHA + FFN + LayerNorm ×2 + 残差"""

    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, L, d_model]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ═══════════════════════════════════════════════
# PTCA-Net 主模型
# ═══════════════════════════════════════════════
class PTCANet(nn.Module):
    """
    PTCA-Net: PTCN + Transformer Encoder

    参数（与 MainSOHModelV2 兼容）:
        in_channels:   输入通道数，默认 3
        seq_len:       时间步长，默认 32
        hidden_dim:    各分支隐藏维度，默认 64
        num_heads:     MHA 头数，默认 8
        cnn_layers:    TCN 每路层数，默认 3
        gru_layers:    Transformer Encoder 层数，默认 1
        gru_dropout:   TCN 内部 dropout，默认 0.1
        se_reduction:  未使用（接口兼容占位）
        dropout:       Transformer/FC 的 dropout，默认 0.2
        output_dim:    输出维度，默认 1
    """

    def __init__(
        self,
        in_channels=3,
        seq_len=32,
        hidden_dim=64,
        num_heads=8,
        cnn_layers=3,
        gru_layers=1,
        gru_dropout=0.1,
        se_reduction=4,
        dropout=0.2,
        output_dim=1,
    ):
        super().__init__()
        d_model = hidden_dim * 3  # 三个 TCN 分支拼接后维度

        # ── 1×1 升维 ──
        self.lift = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)

        # ── 并行 TCN ──
        self.ptcn = ParallelTCN(hidden_dim, cnn_layers, tcn_dropout=gru_dropout)

        # ── 映射到 Transformer 维度 ──
        self.tcn_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # ── Transformer Encoder ──
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(gru_layers)
        ])

        # ── 全局池化 + 回归头 ──
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
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
        # x: [B, in_channels, N]
        x = self.lift(x)                           # [B, H, N]
        x = self.ptcn(x)                           # [B, 3*H, N]
        x = self.tcn_proj(x)                       # [B, 3*H, N]

        # Transformer 需要 [B, L, D]
        x = x.transpose(1, 2)                      # [B, N, 3*H]
        for block in self.transformer_blocks:
            x = block(x)                           # [B, N, 3*H]

        x = x.mean(dim=1)                          # [B, 3*H]  全局平均池化
        return self.fc(x)                          # [B, 1]


# ═══════════════════════════════════════════════
# 快速测试
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PTCANet(
        in_channels=3, seq_len=32, hidden_dim=64,
        num_heads=8, cnn_layers=2, gru_layers=1,
        gru_dropout=0.1, dropout=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"设备: {device}")
    print(f"总参数量: {n_params:,}")

    x = torch.randn(8, 3, 32).to(device)
    y = model(x)
    print(f"输入: {x.shape} → 输出: {y.shape}")

    y.mean().backward()
    print("反向传播: OK")
