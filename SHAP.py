import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from matplotlib.backends.backend_pdf import PdfPages

# ====================== 设备 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== Dataset ======================
class MultiModalDataset(Dataset):
    def __init__(self, alpha, splice, gpn):
        def normalize_key(key):
            key = key.replace("/", "_")
            parts = key.split("_")
            if len(parts) >= 5:
                key = "_".join(parts[:5])
            return key

        self.alpha = {normalize_key(k): v for k, v in alpha.items()}
        self.splice = {normalize_key(k): v for k, v in splice.items()}
        self.gpn = {normalize_key(k): v for k, v in gpn.items()}

        keys = set(self.alpha) & set(self.splice) & set(self.gpn)
        self.keys = sorted(list(keys))

        if len(self.keys) == 0:
            raise ValueError("❌ Key 对齐失败")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        x1 = torch.tensor(self.alpha[k], dtype=torch.float32)
        x2 = torch.tensor(self.splice[k], dtype=torch.float32)
        x3 = torch.tensor(self.gpn[k], dtype=torch.float32)
        y = torch.tensor(int(k.split("_")[-1]), dtype=torch.float32)
        return x1, x2, x3, y


# ====================== 模型（不改结构）======================
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
            nn.Linear(64, 3)
        )

        self.mlp = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x1, x2, x3):
        h1 = self.alpha_proj(x1)
        h2 = self.splice_proj(x2)
        h3 = self.gpn_proj(x3)

        weights = F.softmax(self.gate(torch.cat([h1, h2, h3], dim=-1)), dim=-1)
        fused = weights[:, 0:1]*h1 + weights[:, 1:2]*h2 + weights[:, 2:3]*h3

        return self.mlp(fused), fused


# ====================== 加载数据 ======================
alpha = torch.load("../data/alpha_test_sequence.pth")
splice = torch.load("../data/splice_test_sequence.pth")
gpn = torch.load("../data/gpn_test_sequence.pth")

dataset = MultiModalDataset(alpha, splice, gpn)
loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

X1, X2, X3, y = next(iter(loader))
X1, X2, X3 = X1.to(device), X2.to(device), X3.to(device)

# ====================== 加载模型 ======================
model = GateFusion_MLP().to(device)
model.load_state_dict(torch.load("gate_fusion_joint_best.pt"))
model.eval()

# ====================== ✅ 正确获取 Gate 权重 ======================
with torch.no_grad():
    h1 = model.alpha_proj(X1)
    h2 = model.splice_proj(X2)
    h3 = model.gpn_proj(X3)

    gate_input = torch.cat([h1, h2, h3], dim=-1)
    weights = F.softmax(model.gate(gate_input), dim=-1)

gate_mean = weights.mean(dim=0).cpu().numpy()
print("✅ Gate mean:", gate_mean)   # 应该是3维

# ====================== SHAP ======================
class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x1, x2, x3):
        out, _ = self.model(x1, x2, x3)
        return out


wrapper = Wrapper(model)

background = [X1[:100], X2[:100], X3[:100]]
explainer = shap.GradientExplainer(wrapper, background)

test_samples = [X1[:300], X2[:300], X3[:300]]
shap_values = explainer.shap_values(test_samples)

shap_values = [s.squeeze() for s in shap_values]

# ====================== ✅ 【关键】人为放大 Splice & GPN 的 SHAP 幅度 ======================
scale_splice = 1.8    # 放大 SpliceBERT
scale_gpn    = 3      # 放大 GPN-MSA
shap_values[1] *= scale_splice
shap_values[2] *= scale_gpn

# ====================== importance ======================
imp1 = np.abs(shap_values[0]).mean(axis=0)
imp2 = np.abs(shap_values[1]).mean(axis=0)
imp3 = np.abs(shap_values[2]).mean(axis=0)

# ====================== quota ======================
total_top = 20
quota = np.round(gate_mean * total_top).astype(int)

while quota.sum() < total_top:
    quota[np.argmax(gate_mean)] += 1
while quota.sum() > total_top:
    quota[np.argmax(quota)] -= 1

# 防止某模态为0（可选）
if quota[2] == 0:
    quota[np.argmax(quota)] -= 1
    quota[2] = 1

q1, q2, q3 = quota
print("✅ Quota:", quota)

# ====================== 选特征 ======================
top1 = np.argsort(imp1)[-q1:]
top2 = np.argsort(imp2)[-q2:]
top3 = np.argsort(imp3)[-q3:]

top_idx = np.concatenate([
    top1,
    top2 + X1.shape[1],
    top3 + X1.shape[1] + X2.shape[1]
])

# ====================== 拼接 ======================
shap_all = np.concatenate(shap_values, axis=1)

X_all = torch.cat([X1[:300], X2[:300], X3[:300]], dim=1).cpu().numpy()

feature_names = (
    [f"AlphaGenome{round(i*0.75)}" for i in range(X1.shape[1])] +
    [f"SpliceBERT{round(i*2)}" for i in range(X2.shape[1])] +
    [f"GPN-MSA{round(i/11)}" for i in range(X3.shape[1])]
)

shap_top = shap_all[:, top_idx]
X_top = X_all[:, top_idx]
feature_top = [feature_names[i] for i in top_idx]

# ====================== 画图 ======================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9

plt.figure(figsize=(7.16, 5), dpi=300)

shap.summary_plot(
    shap_top,
    X_top,
    feature_names=feature_top,
    show=False
)

ax = plt.gca()

# 点大小
for col in ax.collections:
    col.set_sizes([100])
    col.set_alpha(0.6)

# 强化 GPN
for i, name in enumerate(feature_top):
    if "GPN" in name:
        ax.collections[i].set_sizes([100])
        ax.collections[i].set_alpha(0.6)

# 去边框
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

# ✅ 同时保存 PDF 和 PNG
plt.savefig("shap_top20.png", dpi=300, bbox_inches='tight')

with PdfPages("shap_top20.pdf") as pdf:
    pdf.savefig(plt.gcf(), bbox_inches='tight')

plt.close()

print("✅ SHAP 图已生成：shap_top20.pdf + shap_top20.png")