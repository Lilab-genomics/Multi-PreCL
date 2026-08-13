from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import numpy as np
from torch.utils.data import Dataset

# =========================
# 1. SpliceBERT 特征加载类
# =========================
class SpliceBERTDataset(Dataset):
    def __init__(self, pth_path, max_len=512):
        """
        严格复刻 load_pth_features 行为
        """
        features = []
        labels = []
        keys = []
        feature_dict = torch.load(pth_path, map_location="cpu")

        for key in feature_dict.keys():
            x = feature_dict[key]
            label = int(key[-1])

            L = x.shape[0]

            # ===== 完全复刻 padding / crop =====
            if L > max_len:
                start = (L - max_len) // 2
                x = x[start:start + max_len]
            elif L < max_len:
                pad_len = max_len - L
                pad = torch.zeros((pad_len, x.shape[1]), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)

            features.append(x.numpy())
            labels.append(label)
            keys.append(key)
        # ===== 完全复刻 numpy =====
        self.X = np.asarray(features)
        self.y = np.asarray(labels)
        self.keys = keys

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]#, self.keys[idx]

# =========================
# 2. GPN-MSA 特征加载类
# =========================
class GPNMSADataset(Dataset):
    def __init__(self, pth_path, max_len=128):
        """
        - 一次性加载
        - padding / crop
        - 返回 numpy（供后续 scaler 使用）
        """
        features = []
        labels = []
        keys = []

        feature_dict = torch.load(pth_path, map_location="cpu")

        for key in feature_dict.keys():
            x = feature_dict[key]   # (L, 768)
            label = int(key[-1])

            L = x.shape[0]

            # ===== 完全复刻 padding / crop =====
            if L > max_len:
                start = (L - max_len) // 2
                x = x[start:start + max_len]
            elif L < max_len:
                pad_len = max_len - L
                pad = torch.zeros((pad_len, x.shape[1]), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)

            features.append(x.numpy())
            labels.append(label)
            keys.append(key)
        self.X = np.asarray(features)
        self.y = np.asarray(labels)
        self.keys = keys


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]#, self.keys[idx]


# =========================
# 3. AlphaGenome 特征加载类
# =========================
class AlphaGenomeDataset(Dataset):
    def __init__(self, pth_path, max_len=128):
        features = []
        labels = []
        keys = []

        feature_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

        for key in feature_dict.keys():
            x = feature_dict[key]
            label = int(key[-1])

            L = x.shape[0]

            if L > max_len:
                start = (L - max_len) // 2
                x = x[start:start + max_len]
            elif L < max_len:
                pad_len = max_len - L
                pad = torch.zeros((pad_len, x.shape[1]), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)

            features.append(x.numpy())
            labels.append(label)
            keys.append(key)

        self.X = np.asarray(features)
        self.y = np.asarray(labels)
        self.keys = keys

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultiModalDataset(Dataset):
    def __init__(self, alpha, splice, gpn, verbose=True):
        def normalize_key(key):
            """
            统一 key:
            chr1_123_A/T_1 → chr1_123_A_T_1
            """
            key = key.replace("/", "_")
            parts = key.split("_")

            # 保证结构：CHROM_POS_REF_ALT_LABEL
            if len(parts) >= 5:
                key = "_".join(parts[:5])

            return key

        # =========================
        # 2. 构建 normalized dict（关键）
        # =========================
        self.alpha = {}
        self.splice = {}
        self.gpn = {}

        for k, v in alpha.items():
            nk = normalize_key(k)
            self.alpha[nk] = v

        for k, v in splice.items():
            nk = normalize_key(k)
            self.splice[nk] = v

        for k, v in gpn.items():
            nk = normalize_key(k)
            self.gpn[nk] = v

        # =========================
        # 3. 计算交集
        # =========================
        a_keys = set(self.alpha.keys())
        s_keys = set(self.splice.keys())
        g_keys = set(self.gpn.keys())

        keys = a_keys & s_keys & g_keys
        self.keys = sorted(list(keys))

        # 安全检查
        if len(self.keys) == 0:
            print("\n❌ ERROR: No aligned samples!")
            print("Alpha example:", list(alpha.keys())[:3])
            print("Splice example:", list(splice.keys())[:3])
            print("GPN example:", list(gpn.keys())[:3])
            raise ValueError("Key 对齐失败，请检查 key 格式")

    def __len__(self):
        return len(self.keys)

    def parse_label(self, key):
        parts = key.split("_")

        try:
            label = int(parts[-1])
        except:
            raise ValueError(f"❌ label 解析失败: {key}")

        return label

    # =========================
    # 8. 获取样本
    # =========================
    def __getitem__(self, idx):
        _id = self.keys[idx]

        x_alpha = torch.as_tensor(self.alpha[_id], dtype=torch.float32)
        x_splice = torch.as_tensor(self.splice[_id], dtype=torch.float32)
        x_gpn = torch.as_tensor(self.gpn[_id], dtype=torch.float32)

        y = torch.tensor(self.parse_label(_id), dtype=torch.float32)

        return x_alpha, x_splice, x_gpn, y










