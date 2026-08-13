import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================
# 外求和版本
# ==============================================
class SupConLoss_Out(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        # 归一化
        features = F.normalize(features, dim=-1)
        sim = torch.matmul(features, features.T) / self.t
        # 同类掩码
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask = mask - torch.eye(B, device=device)  # 去掉自己
        # 屏蔽自身
        logits_mask = 1 - torch.eye(B, device=device)
        exp_sim = torch.exp(sim) * logits_mask
        # ----------------------
        # 🔥 防 NaN 核心修复
        # ----------------------
        pos_sum = torch.clamp((mask * exp_sim).sum(), min=1e-8)
        all_sum = torch.clamp(exp_sim.sum(), min=1e-8)

        loss = -torch.log(pos_sum / all_sum)
        return loss

# ==============================================
# 内求和版本
# ==============================================
class SupConLoss_In(nn.Module):#分母不包含锚样本
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]

        # 归一化
        features = F.normalize(features, dim=-1)

        # 相似度矩阵
        sim = torch.matmul(features, features.T) / self.t

        # 同类掩码
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask = mask - torch.eye(B, device=device)  # 去掉自己

        # 去掉自身
        logits_mask = 1 - torch.eye(B, device=device)
        exp_sim = torch.exp(sim) * logits_mask

        # 核心：每个样本自己的正样本 【内求和】
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # 【外平均】
        return loss.mean()



class SupConLoss_Euclidean(nn.Module): #分母包含锚样本
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]

        # 归一化（可选）
        features = F.normalize(features, dim=-1)

        # ===== 欧式距离 =====
        sq = torch.sum(features ** 2, dim=1, keepdim=True)
        dist2 = sq + sq.T - 2 * torch.matmul(features, features.T)
        dist2 = torch.clamp(dist2, min=1e-12)

        # ===== 相似度 =====
        sim = -dist2 / self.t

        # ===== 正样本 mask（仍然去掉自己）=====
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask = mask - torch.eye(B, device=device)

        # ❗关键修改：分母不再去掉自己
        exp_sim = torch.exp(sim)

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

        loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        return loss.mean()

class TripletLossBatch(nn.Module):
    def __init__(self, margin=1.0, normalize=True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(self, features, labels):
        if self.normalize:
            features = F.normalize(features, dim=-1)

        B = features.size(0)

        # 距离矩阵
        dist = torch.cdist(features, features, p=2)  # (B, B)

        loss = 0.0
        triplet_count = 0

        for i in range(B):
            pos_mask = (labels == labels[i])
            neg_mask = (labels != labels[i])

            pos_mask[i] = False  # 去掉自己

            pos_dist = dist[i][pos_mask]
            neg_dist = dist[i][neg_mask]

            if len(pos_dist) == 0 or len(neg_dist) == 0:
                continue

            # 所有组合
            for dp in pos_dist:
                for dn in neg_dist:
                    loss += F.relu(dp - dn + self.margin)
                    triplet_count += 1

        if triplet_count == 0:
            return torch.tensor(0.0, device=features.device)

        return loss / triplet_count


class SupConLoss_woCompetition(nn.Module):
    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.t = temperature
        self.normalize = normalize

    def forward(self, features, labels):
        device = features.device
        B = features.size(0)

        if self.normalize:
            features = F.normalize(features, dim=-1)

        # 相似度矩阵
        sim = torch.matmul(features, features.T) / self.t  # (B,B)

        loss = 0.0
        valid_count = 0

        for i in range(B):
            pos_mask = (labels == labels[i])
            neg_mask = (labels != labels[i])

            pos_mask[i] = False  # 去掉自己

            pos_idx = torch.where(pos_mask)[0]
            neg_idx = torch.where(neg_mask)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            # 负样本部分（对所有 p 共用）
            neg_sim = sim[i][neg_idx]  # (N_neg,)
            exp_neg = torch.exp(neg_sim).sum()

            # 对每个正样本单独算一个分式
            for p in pos_idx:
                pos_sim = sim[i, p]
                exp_pos = torch.exp(pos_sim)

                denom = exp_pos + exp_neg
                loss_i_p = -torch.log(exp_pos / denom)

                loss += loss_i_p
                valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device)

        return loss / valid_count


# ==============================================
# 🔥 SupConLoss_FocalRank（已去掉无效的 margin 参数）
# ==============================================
# 设计思路：
#   创新1 - Focal 正样本加权：正样本被负样本"竞争过"的（概率低），获得更大权重
#   创新2 - 困难负样本加权：分母中，与锚样本相似度越高的负样本，权重越大
#
# 效果：给难分样本更多关注（Focal 思想），困难负样本被迫优先推开
# ==============================================

class SupConLoss_FocalRank(nn.Module):
    """
    融合 Focal Loss 思想 + 困难负样本加权的监督对比损失

    参数：
        temperature:  基础温度，控制相似度的放缩（默认1.0）
        gamma:        Focal 权重指数，越大则难分正样本权重越高（默认2.0）
                      设0则退化为无Focal加权
        tau_hard:     困难负样本的加权温度，越小则困难负样本权重越集中（默认0.05）
                      设很大（如100）则退化为等权
    """
    def __init__(self, temperature=1.0, gamma=2.0, tau_hard=0.05):
        super().__init__()
        self.t = temperature
        self.gamma = gamma
        self.tau_hard = tau_hard

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]

        # 1. 归一化 + 相似度矩阵
        features = F.normalize(features, dim=-1)
        sim = torch.matmul(features, features.T) / self.t       # [B, B]

        # 2. 构建 bool 掩码
        same_class = (labels.unsqueeze(1) == labels.unsqueeze(0))  # [B, B]
        self_mask = torch.eye(B, dtype=torch.bool, device=device)

        # 正样本掩码（去掉自己）
        pos_mask = same_class & ~self_mask                      # [B, B]

        # ==============================================
        # 困难负样本加权因子（每个锚样本独立计算）
        # ==============================================
        neg_mask = ~same_class                                # [B, B]
        # 将非负样本位置的相似度设为 -inf，让 softmax 忽略
        masked_sim_for_hard = sim.clone()
        masked_sim_for_hard[~neg_mask] = float('-inf')        # 只保留负样本位置
        # 再排除自己（自己的相似度=1/t，会被exp放大，必须排除）
        masked_sim_for_hard[self_mask] = float('-inf')

        # 对每个锚样本 i，在负样本上做 softmax
        hard_weights = F.softmax(masked_sim_for_hard / self.tau_hard, dim=1)  # [B, B]

        # ==============================================
        # 构建加权分母
        # ==============================================
        # 默认权重全1
        denom_weights = torch.ones(B, B, device=device)       # [B, B]
        # 负样本位置替换为困难度权重
        denom_weights[neg_mask] = hard_weights[neg_mask]
        # 排除自己
        denom_weights[self_mask] = 0.0

        exp_sim = torch.exp(sim)                               # [B, B]
        denominator = (denom_weights * exp_sim).sum(dim=1, keepdim=True)  # [B, 1]
        denominator = torch.clamp(denominator, min=1e-12)

        # ==============================================
        # log_prob[i,j] = log( exp(sim[i,j]) / denominator[i] )
        # ==============================================
        log_prob = sim - torch.log(denominator)                # [B, B]

        # ==============================================
        # Focal 正样本加权
        # ==============================================
        prob = exp_sim / denominator                           # [B, B]
        focal_weight = (1.0 - prob).clamp(min=1e-8).pow(self.gamma)  # [B, B]

        # ==============================================
        # 组装最终损失
        # ==============================================
        pos_count = pos_mask.sum(dim=1)                        # [B]
        valid = pos_count > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=features.dtype)

        # 加权求和（Focal权重 × log_prob，仅正样本位置）
        weighted_log_prob = focal_weight * log_prob            # [B, B]
        loss_per_sample = -(pos_mask.float() * weighted_log_prob).sum(dim=1) / (pos_count + 1e-12)

        loss = loss_per_sample[valid].mean()

        # NaN 保护
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, dtype=features.dtype)

        return loss


# ==============================================
# SupConLoss_SigmoidFocal：逐对 Sigmoid + 二值 Focal（无 softmax 分母）
# ==============================================
# 思路：
#   - 不用 exp/softmax 归一化分母，每个 (i,j) 独立二元目标，消除「同类挤一个分母」的竞争。
#   - 正样本对 (i,j)：希望 σ(s_ij)→1，使用 y=1 的 focal：-(1-p)^γ log(p)。
#   - 负样本对 (i,k)：希望 σ(s_ik)→0，使用 y=0 的 focal：-p^γ log(1-p)。
#   - γ=0 时退化为带 logits 的标准 BCE（逐对）。
# ==============================================

class SupConLoss_SigmoidFocal(nn.Module):
    """
    逐对 Sigmoid 监督对比 + 二值 Focal 难分权重（无类内 softmax 竞争）。

    参数：
        temperature: 相似度缩放 s_ij = (z_i·z_j) / t
        gamma:       Focal 指数；0 则退化为普通 BCEWithLogits 形式的逐对损失
        neg_weight:  负样本项相对正样本项的整体权重（平衡推拉）
    """

    def __init__(self, temperature=0.07, gamma=1.0, neg_weight=0.5):
        super().__init__()
        self.t = temperature
        self.gamma = gamma
        self.neg_weight = neg_weight

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        features = F.normalize(features, dim=-1)
        sim = torch.matmul(features, features.T) / self.t

        same_class = (labels.unsqueeze(1) == labels.unsqueeze(0))
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = same_class & ~self_mask
        neg_mask = ~same_class

        p = torch.sigmoid(sim)
        eps = 1e-8
        # y=1 focal：-(1-p)^γ log(p)；log(p)=logsigmoid(sim)
        focal_pos = (1.0 - p).clamp(min=eps).pow(self.gamma) * (-F.logsigmoid(sim))
        # y=0 focal：-p^γ log(1-p)；log(1-p)=logsigmoid(-sim)
        focal_neg = p.clamp(min=eps).pow(self.gamma) * (-F.logsigmoid(-sim))

        pos_count = pos_mask.sum(dim=1)
        neg_count = neg_mask.sum(dim=1)
        valid = (pos_count > 0) & (neg_count > 0)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=features.dtype)

        loss_pos = (focal_pos * pos_mask.float()).sum(dim=1) / (pos_count.float() + 1e-12)
        loss_neg = (focal_neg * neg_mask.float()).sum(dim=1) / (neg_count.float() + 1e-12)
        loss_per_anchor = loss_pos + self.neg_weight * loss_neg
        loss = loss_per_anchor[valid].mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, dtype=features.dtype)

        return loss

class SupConLoss_Sigmoid(nn.Module):
    """
    Sigmoid-based Supervised Contrastive Loss

    核心思想：
        用 pairwise sigmoid 替代 InfoNCE softmax

    特点：
        1. 不存在 positive-positive competition
        2. 不存在 global softmax denominator
        3. 每个 pair 独立优化
        4. 更适合 fine-grained representation learning

    参数：
        temperature:
            相似度缩放温度

        bias:
            decision boundary bias
            类似 SCS-SupCon 中的 learnable b
    """

    def __init__(
        self,
        temperature=0.09,
        bias=0.0,
    ):
        super().__init__()

        self.t = temperature
        self.bias = bias

    def forward(self, features, labels):

        device = features.device
        B = features.shape[0]

        # =====================================================
        # feature normalize
        # =====================================================
        features = F.normalize(features, dim=-1)

        # =====================================================
        # cosine similarity
        # =====================================================
        sim = torch.matmul(features, features.T)  # [B,B]

        # temperature scaling
        sim = sim / self.t

        # decision boundary shifting
        logits = sim - self.bias

        # =====================================================
        # masks
        # =====================================================
        labels = labels.contiguous().view(-1, 1)

        same_class = torch.eq(labels, labels.T).to(device)

        self_mask = torch.eye(B, dtype=torch.bool, device=device)

        pos_mask = same_class & (~self_mask)

        neg_mask = (~same_class)

        # =====================================================
        # sigmoid probability
        # =====================================================
        prob = torch.sigmoid(logits)

        # =====================================================
        # positive pairs
        # 希望 prob -> 1
        # =====================================================
        pos_loss = -torch.log(
            prob.clamp(min=1e-8)
        )

        pos_loss = pos_loss * pos_mask.float()

        # =====================================================
        # negative pairs
        # 希望 prob -> 0
        # =====================================================
        neg_loss = -torch.log(
            (1.0 - prob).clamp(min=1e-8)
        )

        neg_loss = neg_loss * neg_mask.float()

        # =====================================================
        # aggregate
        # =====================================================
        pos_count = pos_mask.sum(dim=1)

        valid = pos_count > 0

        if valid.sum() == 0:
            return torch.tensor(
                0.0,
                device=device,
                dtype=features.dtype
            )

        pos_loss = pos_loss.sum(dim=1) / (pos_count + 1e-12)

        neg_loss = neg_loss.sum(dim=1) / (
            neg_mask.sum(dim=1) + 1e-12
        )

        loss_per_sample = pos_loss + neg_loss

        loss = loss_per_sample[valid].mean()

        # =====================================================
        # nan protection
        # =====================================================
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(
                0.0,
                device=device,
                dtype=features.dtype
            )

        return loss