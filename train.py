import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from models import MLPClassifier
from loss_functions import (
    SupConLoss_In, SupConLoss_Out,
    SupConLoss_Euclidean, TripletLossBatch, SupConLoss_woCompetition,
    SupConLoss_FocalRank,SupConLoss_SigmoidFocal,SupConLoss_Sigmoid
)
import pandas as pd
import math





# ==========================================
# BCEWithLogitsLoss
# ==========================================
def train_BCEWithLogitsLoss(
    model,
    train_loader,
    test_loader,
    optimizer,
    device,
):
    epochs=50
    patience=10
    best_auc=0.0
    save_path = "./checkpoint/gate_fusion_BCEWithLogits_best.pt"
    criterion = nn.BCEWithLogitsLoss()
    counter = 0  # 必须初始化

    # ======================
    # 🔥 自动从训练集里拆分出验证集（你要的核心！）
    # ======================
    train_dataset = train_loader.dataset
    val_size = int(0.1 * len(train_dataset))  # 10% 作为验证
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # 重新构建新的 train/val loader
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=train_loader.batch_size, shuffle=False)

    # ======================
    # 训练循环（完全不变）
    # ======================
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x_alpha, x_splice, x_gpn, y in train_loader:
            x_alpha = x_alpha.to(device)
            x_splice = x_splice.to(device)
            x_gpn = x_gpn.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logit, fused = model(x_alpha, x_splice, x_gpn)
            loss = criterion(logit.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ======================
        # 验证
        # ======================
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x_alpha, x_splice, x_gpn, y in val_loader:
                x_alpha = x_alpha.to(device)
                x_splice = x_splice.to(device)
                x_gpn = x_gpn.to(device)
                logit, _ = model(x_alpha, x_splice, x_gpn)
                prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                preds.extend(prob)
                labels.extend(y.cpu().numpy())

        val_auc = roc_auc_score(labels, preds)
        print(f"Epoch {epoch+1:2d} | Loss={total_loss:.4f} | Val AUC={val_auc:.4f}")

        # ======================
        # 早停
        # ======================
        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), save_path)
            print("✔ 最优模型已保存")
        else:
            counter += 1
            print(f"早停计数: {counter}/{patience}")
            if counter >= patience:
                print("触发早停")
                break

    # ======================
    # 测试集评估
    # ======================
    print("\n===== 测试集性能 =====")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x_alpha, x_splice, x_gpn, y in test_loader:
            x_alpha = x_alpha.to(device)
            x_splice = x_splice.to(device)
            x_gpn = x_gpn.to(device)
            logit, _ = model(x_alpha, x_splice, x_gpn)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            preds.extend(prob)
            labels.extend(y.cpu().numpy())

    y_true = np.array(labels)
    y_pred = (np.array(preds) >= 0.5).astype(int)
    y_prob = np.array(preds)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"ACC       : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"AUPR      : {aupr:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

def extract_features(model, loader, device):
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for x_alpha, x_splice, x_gpn, y in loader:
            x_alpha = x_alpha.to(device)
            x_splice = x_splice.to(device)
            x_gpn = x_gpn.to(device)

            fused = model(x_alpha, x_splice, x_gpn)

            features.append(fused.cpu())
            labels.append(y)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels

def train_SupCon_decoupling(
    model,
    train_loader,
    optimizer,
    device,
    temperature=0.07
):
    criterion = SupConLoss_Sigmoid(temperature)

    # ===== 早停参数（写死在函数内部）=====
    epochs = 50
    patience = 5
    min_delta = 1e-3
    ema_alpha = 0.3
    save_path = "./checkpoint/fusion_contrast.pt"

    best_loss = float("inf")
    ema_loss = None
    counter = 0

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x_alpha, x_splice, x_gpn, y in train_loader:
            x_alpha = x_alpha.to(device)
            x_splice = x_splice.to(device)
            x_gpn = x_gpn.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            fused = model(x_alpha, x_splice, x_gpn)

            fused = F.normalize(fused, dim=-1)

            loss = criterion(fused, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== EMA 平滑 =====
        if ema_loss is None:
            ema_loss = avg_loss
        else:
            ema_loss = ema_alpha * avg_loss + (1 - ema_alpha) * ema_loss

        print(f"Epoch {epoch+1} | Loss={avg_loss:.4f} | EMA={ema_loss:.4f}")

        # ===== 早停判断 =====
        if ema_loss < best_loss - min_delta:
            best_loss = ema_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
            print("✔ 保存最优 Fusion")
        else:
            counter += 1
            print(f"早停计数: {counter}/{patience}")

            if counter >= patience:
                print("⛔ 触发早停（loss 已收敛）")
                break

    print("✔ Fusion 训练完成")



def train_mlp_classifier(
    train_feat,
    train_label,
    test_feat,
    test_label,
    device,
    val_ratio=0.2,
    batch_size=64,
    lr=1e-3,
    epochs=50,
    patience=10
):
    # ===== 1. 构建 Dataset =====
    dataset = TensorDataset(train_feat, train_label)

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(test_feat, test_label)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ===== 2. 模型 =====
    model = MLPClassifier(in_dim=train_feat.shape[1]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    counter = 0

    # ===== 3. 训练 =====
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            logit = model(x).squeeze()
            loss = criterion(logit, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== 验证 =====
        model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)

                logit = model(x).squeeze()
                prob = torch.sigmoid(logit).cpu().numpy()

                preds.extend(prob)
                labels.extend(y.numpy())

        val_auc = roc_auc_score(labels, preds)

        print(f"Epoch {epoch+1:2d} | Loss={avg_loss:.4f} | Val AUC={val_auc:.4f}")

        # ===== Early Stopping =====
        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), "./checkpoint/mlp_best.pt")
            print("✔ 保存最优 MLP")
        else:
            counter += 1
            print(f"早停计数: {counter}/{patience}")

            if counter >= patience:
                print("⛔ 触发早停")
                break

    # ===== 4. 测试 =====
    print("\n===== 测试集性能 =====")

    model.load_state_dict(torch.load("./checkpoint/mlp_best.pt", map_location=device))
    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            logit = model(x).squeeze()
            prob = torch.sigmoid(logit).cpu().numpy()

            preds.extend(prob)
            labels.extend(y.numpy())

    y_true = np.array(labels)
    y_prob = np.array(preds)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"ACC       : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"AUPR      : {aupr:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")


def train_SupCon_BCEWithLogitsLoss(
        model,
        train_loader,
        test_loader,
        optimizer,
        device,
        temperature=0.5,
        epochs=50,
        patience=10,
        best_auc=0.0
):
    save_path = "./checkpoint/gate_fusion_joint_best.pt"

    #SupConLoss_YJJ, SupConLoss_In, SupConLoss_Out, SupConLoss_ZYR, SupConLoss_Euclidean, TripletLossBatch, SupConLoss_woCompetition
    # 两个损失
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_con = SupConLoss_In(temperature)

    counter = 0

    train_dataset = train_loader.dataset
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=train_loader.batch_size, shuffle=False)

    # ======================
    # 训练循环
    # ======================
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_con = 0.0  # 对比损失

        for x_alpha, x_splice, x_gpn, y in train_loader:
            x_alpha = x_alpha.to(device)
            x_splice = x_splice.to(device)
            x_gpn = x_gpn.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            logit, fused = model(x_alpha, x_splice, x_gpn)

            loss_bce = criterion_bce(logit.view(-1), y)
            loss_con = criterion_con(fused, y)
            loss = loss_bce + loss_con

            # 反向
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_con += loss_con.item()

        # 平均
        avg_loss = total_loss / len(train_loader)
        avg_bce = total_bce / len(train_loader)
        avg_con = total_con / len(train_loader)

        # ======================
        # 验证
        # ======================
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x_alpha, x_splice, x_gpn, y in val_loader:
                x_alpha = x_alpha.to(device)
                x_splice = x_splice.to(device)
                x_gpn = x_gpn.to(device)
                logit, _ = model(x_alpha, x_splice, x_gpn)
                prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                prob = np.nan_to_num(prob, nan=0.5)
                preds.extend(prob)
                labels.extend(y.cpu().numpy())

        val_auc = roc_auc_score(labels, preds)

        # ======================
        # ✅ 每轮输出三个 Loss
        # ======================
        print(
            f"Epoch {epoch + 1:2d} | Loss={avg_loss:.4f} | BCE={avg_bce:.4f} | Con={avg_con:.4f} | Val AUC={val_auc:.4f}")

        # ======================
        # 早停
        # ======================
        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), save_path)
            print("✔ 最优模型已保存")
        else:
            counter += 1
            print(f"早停计数: {counter}/{patience}")
            if counter >= patience:
                print("触发早停")
                break

    # ======================
    # 测试集
    # ======================
    print("\n===== 测试集性能 =====")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x_alpha, x_splice, x_gpn, y in test_loader:
            x_alpha = x_alpha.to(device)
            x_splice = x_splice.to(device)
            x_gpn = x_gpn.to(device)
            logit, _ = model(x_alpha, x_splice, x_gpn)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            prob = np.nan_to_num(prob, nan=0.5)
            preds.extend(prob)
            labels.extend(y.cpu().numpy())

    y_true = np.array(labels)
    y_pred = (np.array(preds) >= 0.5).astype(int)
    y_prob = np.array(preds)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"ACC       : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"AUPR      : {aupr:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")


def train_SupCon_BCEWithLogitsLoss_5fold(
        model,
        full_train_dataset,
        test_loader,
        optimizer,
        device,
        temperature=1.0,
        epochs=50,
        patience=10,
):

    os.makedirs("./checkpoint", exist_ok=True)
    metric_names = ["ACC", "AUC", "AUPR", "MCC", "Precision", "Recall", "F1"]
    all_fold_results = []

    # 5折划分
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_dataset)):
        print(f"\n=====================================")
        print(f"               第 {fold + 1} 折")
        print(f"=====================================")

        # 折内数据划分
        train_subset = Subset(full_train_dataset, train_idx)
        val_subset = Subset(full_train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        model = model.__class__().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_auc = 0.0
        counter = 0
        save_path = f"./checkpoint/gate_fusion_joint_fold{fold + 1}_best.pt"

        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_con = SupConLoss_In(temperature)

        # 训练
        for epoch in range(epochs):
            model.train()
            total_loss = total_bce = total_con = 0.0
            for x_alpha, x_splice, x_gpn, y in train_loader:
                x_alpha = x_alpha.to(device)
                x_splice = x_splice.to(device)
                x_gpn = x_gpn.to(device)
                y = y.to(device).float()

                optimizer.zero_grad()
                logit, fused = model(x_alpha, x_splice, x_gpn)

                loss_bce = criterion_bce(logit.view(-1), y)
                loss_con = criterion_con(fused, y)
                loss = loss_bce + loss_con

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_bce += loss_bce.item()
                total_con += loss_con.item()

            avg_loss = total_loss / len(train_loader)
            avg_bce = total_bce / len(train_loader)
            avg_con = total_con / len(train_loader)

            # 验证
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for x_alpha, x_splice, x_gpn, y in val_loader:
                    x_alpha = x_alpha.to(device)
                    x_splice = x_splice.to(device)
                    x_gpn = x_gpn.to(device)
                    logit, _ = model(x_alpha, x_splice, x_gpn)
                    prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                    prob = np.nan_to_num(prob, nan=0.5)
                    preds.extend(prob)
                    labels.extend(y.cpu().numpy())

            val_auc = roc_auc_score(labels, preds)

            print(
                f"F{fold + 1} E{epoch + 1:2d} | Loss={avg_loss:.4f} BCE={avg_bce:.4f} Con={avg_con:.4f} | Val AUC={val_auc:.4f}")

            # 早停
            if val_auc > best_auc:
                best_auc = val_auc
                counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= patience:
                    print(f"Fold {fold + 1} 触发早停")
                    break


        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x_alpha, x_splice, x_gpn, y in test_loader:
                x_alpha = x_alpha.to(device)
                x_splice = x_splice.to(device)
                x_gpn = x_gpn.to(device)
                logit, _ = model(x_alpha, x_splice, x_gpn)
                prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                prob = np.nan_to_num(prob, nan=0.5)
                preds.extend(prob)
                labels.extend(y.cpu().numpy())

        y_true = np.array(labels)
        y_prob = np.array(preds)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "ACC": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob),
            "AUPR": average_precision_score(y_true, y_prob),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0)
        }
        all_fold_results.append(metrics)

        # 打印当前折结果
        print(f"\n[第 {fold + 1} 折 测试集结果]")
        for k, v in metrics.items():
            print(f"{k:10s}: {v:.4f}")

    # ======================
    # 5 折汇总输出
    # ======================
    print("\n\n" + "=" * 60)
    print("           5 折交叉验证 最终汇总结果")
    print("=" * 60)

    # 收集每折数据
    fold_values = {k: [] for k in metric_names}
    for i, res in enumerate(all_fold_results):
        print(f"\n[第 {i + 1} 折]")
        for k in metric_names:
            fold_values[k].append(res[k])
            print(f"  {k:10s}: {res[k]:.4f}")

    # 均值 ± 标准差
    print("\n" + "-" * 60)
    print("5 折 均值 ± 标准差")
    print("-" * 60)
    for k in metric_names:
        mean_v = np.mean(fold_values[k])
        std_v = np.std(fold_values[k])
        print(f"{k:10s}: {mean_v:.4f} ± {std_v:.4f}")

    # 每个指标的 5 个原始值
    print("\n" + "-" * 60)
    print("每个指标的 5 折原始值")
    print("-" * 60)
    for k in metric_names:
        print(f"{k:10s}: {[round(x, 4) for x in fold_values[k]]}")

    return all_fold_results

def train_SupCon_BCEWithLogitsLoss_fixed_epoch(
        model,
        train_loader,
        test_loader,
        optimizer,
        device,
        fixed_epochs,
        temperature=0.5,
):
    save_path = "./checkpoint/gate_fusion_joint_fixed_epoch_best.pt"

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_con = SupConLoss_Sigmoid(temperature)

    # ======================
    # 训练
    # ======================
    for epoch in range(fixed_epochs):
        model.train()

        total_loss = 0.0
        total_bce = 0.0
        total_con = 0.0

        for x_alpha, x_splice, x_gpn, y in train_loader:
            x_alpha = x_alpha.to(device)
            x_splice = x_splice.to(device)
            x_gpn = x_gpn.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()

            logit, fused = model(x_alpha, x_splice, x_gpn)

            loss_bce = criterion_bce(logit.view(-1), y)
            loss_con = criterion_con(fused, y)
            loss = loss_bce + loss_con * 0.225

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_con += loss_con.item()

        avg_loss = total_loss / len(train_loader)
        avg_bce = total_bce / len(train_loader)
        avg_con = total_con / len(train_loader)

        print(f"Epoch {epoch + 1:2d} | Loss={avg_loss:.4f} | BCE={avg_bce:.4f} | Con={avg_con:.4f}")

    # ======================
    # 保存模型
    # ======================
    torch.save(model.state_dict(), save_path)

    # ======================
    # 测试评估 + 保存 fused
    # ======================
    print("\n===== 测试集性能（固定 epoch 训练） =====")

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    preds, labels = [], []
    fused_dict = {}

    idx_ptr = 0  # 样本索引

    with torch.no_grad():
        for x_alpha, x_splice, x_gpn, y in test_loader:
            x_alpha = x_alpha.to(device)
            x_splice = x_splice.to(device)
            x_gpn = x_gpn.to(device)

            logit, fused = model(x_alpha, x_splice, x_gpn)

            prob = torch.sigmoid(logit).view(-1).cpu().numpy()
            prob = np.nan_to_num(prob, nan=0.5)

            preds.extend(prob)
            labels.extend(y.cpu().numpy())

            # ======================
            # 保存 fused 表征
            # ======================
            fused_np = fused.cpu().numpy()
            batch_keys = test_loader.dataset.keys[idx_ptr:idx_ptr + len(fused_np)]

            for i, key in enumerate(batch_keys):
                fused_dict[key] = fused_np[i]  # 用 key 而不是数字！

            idx_ptr += len(fused_np)

    # ======================
    # 保存 fused 文件
    # ======================
    torch.save(fused_dict, "./hidden_layer/test_fusion.pth")
    print("✅ 已保存 fused 表征 -> ./hidden_layer/test_fusion.pth")

    keys = test_loader.dataset.keys
    if len(keys) != len(preds):
        raise ValueError(f"keys 数量 ({len(keys)}) 和 preds 数量 ({len(preds)}) 不一致！")
    df = pd.DataFrame({
        "key": keys,
        "score": preds
    })
    csv_path = "./hidden_layer/test_predictions.csv"
    df.to_csv(csv_path, index=False)

    print(f"✅ 已保存预测分数 -> {csv_path}")
    # ======================
    # 计算指标
    # ======================
    y_true = np.array(labels)
    y_prob = np.array(preds)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"ACC       : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"AUPR      : {aupr:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")



def train_SupCon_BCEWithLogitsLoss_fixed_epoch_search_a(
        model,
        train_loader,
        test_loader,
        optimizer,
        device,
        fixed_epochs,
        temperature=0.5,
        a=1.0,
):
    import numpy as np

    save_path = f"./checkpoint/gate_fusion_joint_fixed_epoch_a_{a:.1f}.pt"

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_con = SupConLoss_Sigmoid(temperature)

    # ======================
    # 训练
    # ======================
    for epoch in range(fixed_epochs):

        model.train()

        total_loss = 0.0
        total_bce = 0.0
        total_con = 0.0

        for x_alpha, x_splice, x_gpn, y in train_loader:

            x_alpha = x_alpha.to(device)
            x_splice = x_splice.to(device)
            x_gpn = x_gpn.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()

            logit, fused = model(x_alpha, x_splice, x_gpn)

            loss_bce = criterion_bce(logit.view(-1), y)
            loss_con = criterion_con(fused, y)

            # ======================
            # SupCon 加权
            # ======================
            loss = loss_bce + loss_con * a

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_con += loss_con.item()

        avg_loss = total_loss / len(train_loader)
        avg_bce = total_bce / len(train_loader)
        avg_con = total_con / len(train_loader)

        print(
            f"[a={a:.4f}] "
            f"Epoch {epoch + 1:2d} | "
            f"Loss={avg_loss:.4f} | "
            f"BCE={avg_bce:.4f} | "
            f"Con={avg_con:.4f}"
        )

    # ======================
    # 保存模型
    # ======================
    torch.save(model.state_dict(), save_path)

    # ======================
    # 测试评估
    # ======================
    print(f"\n===== 测试集性能 (a={a:.4f}) =====")

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    preds, labels = [], []

    with torch.no_grad():

        for x_alpha, x_splice, x_gpn, y in test_loader:

            x_alpha = x_alpha.to(device)
            x_splice = x_splice.to(device)
            x_gpn = x_gpn.to(device)

            logit, fused = model(x_alpha, x_splice, x_gpn)

            prob = torch.sigmoid(logit).view(-1).cpu().numpy()
            prob = np.nan_to_num(prob, nan=0.5)

            preds.extend(prob)
            labels.extend(y.cpu().numpy())

    # ======================
    # 计算指标
    # ======================
    y_true = np.array(labels)
    y_prob = np.array(preds)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"ACC       : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"AUPR      : {aupr:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # ======================
    # return 指标
    # ======================
    return {
        "a": round(a, 4),
        "ACC": acc,
        "AUC": auc,
        "AUPR": aupr,
        "MCC": mcc,
        "Precision": prec,
        "Recall": recall,
        "F1": f1
    }




