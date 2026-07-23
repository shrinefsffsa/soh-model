"""
CNN-Transformer-LSTM: 串行混合模型

基于 Zeng et al., "CNN-Transformer-LSTM-based lithium battery health state prediction",
J. Phys. Conf. Ser. 3096 (2025) 012008.

架构（串行堆叠）:
  输入 [B, 3, N]
    │
    CNN (单层 Conv1d + ReLU) → 局部特征提取
    │
    Transformer Encoder ×2 → 全局依赖建模
    │
    LSTM → 时序平滑
    │
    FC → SOH

接口与 MainSOHModelV2 兼容，可直接替换贝叶斯寻优.py 的 import。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTransLSTM(nn.Module):
    """
    CNN-Transformer-LSTM 串行混合模型

    参数（与 MainSOHModelV2 兼容）:
        in_channels:   输入通道数，默认 3
        seq_len:       时间步长，默认 32
        hidden_dim:    CNN 输出 / Transformer 嵌入维度，默认 64
        num_heads:     MHA 头数，默认 2
        cnn_layers:    未使用（接口兼容）
        gru_layers:    Transformer Encoder 层数，默认 2
        gru_dropout:   LSTM dropout，默认 0.1
        se_reduction:  未使用（接口兼容）
        dropout:       Transformer dropout，默认 0.1
        output_dim:    输出维度，默认 1
    """

    def __init__(
        self,
        in_channels=3,
        seq_len=32,
        hidden_dim=64,
        num_heads=2,
        cnn_layers=3,
        gru_layers=2,
        gru_dropout=0.1,
        se_reduction=4,
        dropout=0.1,
        output_dim=1,
    ):
        super().__init__()
        lstm_hidden = hidden_dim * 2  # LSTM hidden = CNN out × 2

        # ── CNN：单卷积层提取局部特征 ──
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ── 位置编码 + Transformer Encoder ──
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(gru_layers)
        ])

        # ── LSTM：时序平滑 ──
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=gru_dropout,
        )

        # ── 回归头 ──
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, output_dim),
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
        x = self.conv(x)                     # [B, hidden, N]
        x = x.transpose(1, 2)                # [B, N, hidden]
        x = x + self.pos_embed[:, :x.shape[1], :]  # 位置编码

        for block in self.transformer_blocks:
            x = block(x)                     # [B, N, hidden]

        out, _ = self.lstm(x)                # [B, N, lstm_hidden]
        out = out[:, -1, :]                  # 取最后一步 LSTM 输出

        return self.fc(out)                  # [B, 1]


# ═══════════════════════════════════════════════
# 快速测试
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTransLSTM(
        in_channels=3, seq_len=32, hidden_dim=64,
        num_heads=2, gru_layers=2, gru_dropout=0.1, dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"设备: {device}")
    print(f"总参数量: {n_params:,}")

    x = torch.randn(8, 3, 32).to(device)
    y = model(x)
    print(f"输入: {x.shape} → 输出: {y.shape}")

    y.mean().backward()
    print("反向传播: OK")
