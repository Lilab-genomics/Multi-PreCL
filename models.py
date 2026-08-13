import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ======================================================
# TextCNN_block
# ======================================================
class TextCNN_block1(nn.Module):
    def __init__(self, embedding_size_DLM1, n_filters, filter_sizes, output_dim, dropout):
        super(TextCNN_block1, self).__init__()

        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_size_DLM1,
                out_channels=n_filters,
                kernel_size=fs,
                padding='same'
            )
            for fs in filter_sizes
        ])

        self.flat_dim = n_filters * len(filter_sizes) * 16

        self.dropout1 = nn.Dropout(dropout)
        self.Mish1 = nn.Mish()

        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 32),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(32, 8),
            nn.Mish(),
            nn.Linear(8, output_dim)
        )

    def forward(self, x, mode=None):
        # ===== 完全保持原始流程 =====
        x = x.permute(0, 2, 1)

        conved = [self.Mish1(conv(x)) for conv in self.convs1]

        pooled = [
            F.max_pool1d(conv, math.ceil(conv.shape[2] / 16))
            for conv in conved
        ]

        flatten = [p.contiguous().view(p.size(0), -1) for p in pooled]

        cat = self.dropout1(torch.cat(flatten, dim=1))  # (B, 8192)

        # ===== 新增：residue-level =====
        residue_feat = torch.cat(pooled, dim=1)        # (B, 512, 16)
        residue_feat = residue_feat.permute(0, 2, 1)   # (B, 16, 512)

        # ===== 新增模式 =====
        if mode == "feature":
            return cat

        if mode == "both":
            return residue_feat, cat

        # ===== 默认行为（完全等价原版）=====
        return self.fc(cat), cat

# 包装成与原CNN接口一致的模型
class TextCNN(nn.Module):
    def __init__(self, input_dim, sequence_length):
        super(TextCNN, self).__init__()
        # 使用TextCNN_block1，参数适配你的数据
        self.block = TextCNN_block1(
            embedding_size_DLM1=input_dim,  # 768
            n_filters=128,  # 每个卷积核输出通道
            filter_sizes=[3, 5, 7, 9],  # 多尺度卷积核
            output_dim=1,  # 二分类
            dropout=0.3
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mode=None):
        if mode is not None:
            return self.block(x, mode=mode)
        output, _ = self.block(x)
        return self.sigmoid(output)


class DCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

# =========================
# DCNN + MLP (不动)
# =========================
class DCNN_MLP(nn.Module):
    def __init__(self, max_len=128, dim=1536):
        super().__init__()

        self.dcnn = nn.Sequential(
            DCNNBlock(dim, 512, kernel_size=5, dilation=1),
            nn.MaxPool1d(2),
            DCNNBlock(512, 256, kernel_size=5, dilation=2),
            nn.MaxPool1d(2),
            DCNNBlock(256, 128, kernel_size=5, dilation=4),
            nn.MaxPool1d(2),
        )

        dcn_len = max_len // 8

        # 分类头
        self.mlp = nn.Sequential(
            nn.Linear(128 * dcn_len, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x, mode="train"):
        """
        mode:
            'train'   → 输出 logits
            'feature' → 输出 sequence feature
            'both'    → 输出 (residue, sequence)
        """
        x = x.transpose(1, 2)
        x = self.dcnn(x)

        residue_feat = x.permute(0, 2, 1)   # (B, 16, 128)
        sequence_feat = x.flatten(1)        # (B, 2048)

        if mode == "feature":
            return sequence_feat

        if mode == "both":
            return residue_feat, sequence_feat

        return self.mlp(sequence_feat)

class CNN(nn.Module):
    def __init__(self, input_dim, sequence_length):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.3)
        self.flat_dim = 256 * (sequence_length // 4)

        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.dropout3 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, mode="train"):
        """
        mode:
            'train'   → 输出分类结果
            'feature' → 输出 sequence-level feature
            'both'    → 输出 (residue, sequence)
        """
        x = x.permute(0, 2, 1)              # (B, D, L)

        x = self.pool(self.relu(self.conv1(x)))   # (B, 512, 255)
        x = self.dropout1(x)

        x = self.pool(self.relu(self.conv2(x)))   # (B, 256, 127)
        x = self.dropout2(x)

        # ===== 提取两种特征 =====
        residue_feat = x.permute(0, 2, 1)         # (B, 127, 256)

        x = x.view(x.size(0), -1)                 # flatten
        sequence_feat = self.relu(self.fc1(x))    # (B, 256)
        sequence_feat = self.dropout3(sequence_feat)

        # ===== 输出控制 =====
        if mode == "feature":
            return sequence_feat

        if mode == "both":
            return residue_feat, sequence_feat

        # 默认训练
        logits = self.fc2(sequence_feat)
        return torch.sigmoid(logits)


class GateFusion_MLP_earlystop(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha_proj = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.splice_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.gpn_proj = nn.Sequential(
            nn.Linear(8192, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.gate = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出3个权重
        )

        self.mlp = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x_alpha, x_splice, x_gpn):
        h1 = self.alpha_proj(x_alpha)
        h2 = self.splice_proj(x_splice)
        h3 = self.gpn_proj(x_gpn)

        # 计算门控权重
        gate_input = torch.cat([h1, h2, h3], dim=-1)
        weights = F.softmax(self.gate(gate_input), dim=-1)

        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]
        w3 = weights[:, 2:3]

        # 加权融合
        fused = w1 * h1 + w2 * h2 + w3 * h3

        # 预测输出
        logit = self.mlp(fused)

        return logit, fused

class GateFusion_MLP_5floder(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha_proj = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Mish(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.LayerNorm(128)
        )

        self.splice_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.Mish(),
            nn.LayerNorm(128)
        )

        self.gpn_proj = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.Mish(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 128),
            nn.Mish(),
            nn.LayerNorm(128)
        )

        self.gate = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.Mish(),
            nn.Linear(64, 3)  # 输出3个权重
        )

        self.mlp = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, x_alpha, x_splice, x_gpn):
            h1 = self.alpha_proj(x_alpha)
            h2 = self.splice_proj(x_splice)
            h3 = self.gpn_proj(x_gpn)

            gate_input = torch.cat([h1, h2, h3], dim=-1)
            weights = F.softmax(self.gate(gate_input), dim=-1)

            w1 = weights[:, 0:1]
            w2 = weights[:, 1:2]
            w3 = weights[:, 2:3]

            fused = w1 * h1 + w2 * h2 + w3 * h3  # (B,128)
            logit = self.mlp(fused)
            return logit, fused

class GateFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha_proj = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.splice_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.gpn_proj = nn.Sequential(
            nn.Linear(8192, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.gate = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # 输出3个权重
        )


    def forward(self, x_alpha, x_splice, x_gpn):
            h1 = self.alpha_proj(x_alpha)
            h2 = self.splice_proj(x_splice)
            h3 = self.gpn_proj(x_gpn)

            gate_input = torch.cat([h1, h2, h3], dim=-1)
            weights = F.softmax(self.gate(gate_input), dim=-1)

            w1 = weights[:, 0:1]
            w2 = weights[:, 1:2]
            w3 = weights[:, 2:3]

            fused = w1 * h1 + w2 * h2 + w3 * h3  # (B,128)
            return fused

class MLPClassifier(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.mlp(x)



    '''
    mode3 best
    class GateFusion_MLP(nn.Module):
        def __init__(self):
            super().__init__()
    
            self.alpha_proj = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.LayerNorm(128)
            )
    
            self.splice_proj = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.LayerNorm(128)
            )
    
            self.gpn_proj = nn.Sequential(
                nn.Linear(8192, 128),
                nn.ReLU(),
                nn.LayerNorm(128)
            )
    
            self.gate = nn.Sequential(
                nn.Linear(128 * 3, 64),
                nn.ReLU(),
                nn.Linear(64, 3)   # 输出3个权重
            )
    
            self.mlp = nn.Sequential(
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1)
            )
        '''
class GateFusion_MLP_try(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha_proj = nn.Sequential(
            nn.Linear(2048, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )

        self.splice_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )

        self.gpn_proj = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )

        self.gate = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.GELU(),
            nn.Linear(64, 3)  # 输出3个权重
        )

        self.mlp = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, x_alpha, x_splice, x_gpn):
            h1 = self.alpha_proj(x_alpha)
            h2 = self.splice_proj(x_splice)
            h3 = self.gpn_proj(x_gpn)

            gate_input = torch.cat([h1, h2, h3], dim=-1)
            weights = F.softmax(self.gate(gate_input), dim=-1)

            w1 = weights[:, 0:1]
            w2 = weights[:, 1:2]
            w3 = weights[:, 2:3]

            fused = w1 * h1 + w2 * h2 + w3 * h3  # (B,128)
            logit = self.mlp(fused)
            return logit, fused

class TokenCNNEncoder(nn.Module):
    """
    输入:
        x: [B, L, D_in]

    输出:
        h: [B, out_dim]
    """

    def __init__(
        self,
        dim=1536,
        hidden_dim=256,
        out_dim=256,
        num_conv_layers=2,
        dropout=0.1,

    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        conv_blocks = []
        for _ in range(num_conv_layers):
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )

        self.conv = nn.Sequential(*conv_blocks)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: [B, L, D]
        """
        x = self.input_proj(x)             # [B, L, H]
        x = x.transpose(1, 2)              # [B, H, L]
        x = self.conv(x)                   # [B, H, L]

        mean_pool = x.mean(dim=-1)         # [B, H]
        max_pool = x.max(dim=-1).values    # [B, H]

        h = torch.cat([mean_pool, max_pool], dim=-1)
        h = self.out(h)

        return h


class TextCNNEncoder(nn.Module):
    """
    输入:
        x: [B, L, D] = [B, 128, 1536]

    转换:
        x.permute(0, 2, 1) -> [B, D, L] = [B, 1536, 128]

    每个卷积核尺寸 k 会提取相邻 k 个位置的局部模式。
    最后通过 Global Max Pooling 得到每种卷积核的一维表示。
    """
    def __init__(self, dim, out_channels=128, kernel_sizes=None, dropout=0.30):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=dim,
                out_channels=out_channels,
                kernel_size=k,
                padding=k // 2,
            )
            for k in kernel_sizes
        ])

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.output_dim = out_channels * len(kernel_sizes)

    def forward(self, x):
        # x: [B, L, D]
        x = x.permute(0, 2, 1)  # [B, D, L]

        conv_outputs = []

        for conv in self.convs:
            h = conv(x)                     # [B, C, L]
            h = self.activation(h)
            h = torch.max(h, dim=2).values  # [B, C]
            conv_outputs.append(h)

        z = torch.cat(conv_outputs, dim=1)  # [B, C * len(kernel_sizes)]
        z = self.dropout(z)

        return z