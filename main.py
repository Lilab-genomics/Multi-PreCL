import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Data_utils import (
    AlphaGenomeDataset,
    SpliceBERTDataset,
    GPNMSADataset,
    MultiModalDataset,
)
from models import (
    DCNN_MLP,
    CNN,
    TextCNN,
    GateFusion_MLP_earlystop,
    GateFusion,
    GateFusion_MLP_5floder,
    TokenCNNEncoder,
    TextCNNEncoder,
    GateFusion_MLP_try,
)

from train import (
    train_BCEWithLogitsLoss,
    train_SupCon_BCEWithLogitsLoss,
    extract_features,
    train_SupCon_decoupling,
    train_mlp_classifier,
    train_SupCon_BCEWithLogitsLoss_5fold,
    train_SupCon_BCEWithLogitsLoss_fixed_epoch,
    train_SupCon_BCEWithLogitsLoss_fixed_epoch_search_a
)
import random
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_score, recall_score, f1_score
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_alphagenome(
    train_pth="./Dataset/AlphaGenome_train.pth",
    max_len=128,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=50,
    patience=8,
    save_path="./checkpoint/dcnn_best.pt"
):
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(42)

    dataset = AlphaGenomeDataset(train_pth, max_len=max_len)
    X_all, y_all = dataset.X, dataset.y
    num_samples, sequence_length, feature_dim = X_all.shape

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

    model = DCNN_MLP(max_len=sequence_length, dim=feature_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    def compute_auc(y_true, y_prob):
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_prob)

    best_val_auc = 0.0
    early_stop_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_auc = compute_auc(val_labels, val_probs)
        scheduler.step(val_auc)

        print(f"[Epoch {epoch + 1:03d}] Train Loss: {train_loss:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
            print(f"[INFO] Best model saved (Val AUC: {val_auc:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("[INFO] Early stopping triggered.")
                break

    print("✅ AlphaGenome训练完成")

def extract_alphagenome_features():
    dummy_dataset = AlphaGenomeDataset("./Dataset/AlphaGenome_train.pth", max_len=128)
    _, seq_len, feat_dim = dummy_dataset.X.shape
    model = DCNN_MLP(max_len=seq_len, dim=feat_dim).to(DEVICE)
    model.load_state_dict(torch.load("./checkpoint/dcnn_best.pt", weights_only=True))
    model.eval()

    def extract(pth_path, save_prefix):
        dataset = AlphaGenomeDataset(pth_path, max_len=seq_len)
        loader = DataLoader(dataset, batch_size=32)
        keys = dataset.keys
        residue_dict = {}
        sequence_dict = {}
        idx = 0

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(DEVICE, dtype=torch.float32)  # ✅ 强制float32
                residue_feat, sequence_feat = model(x, mode="both")
                residue_feat = residue_feat.cpu().numpy()
                sequence_feat = sequence_feat.cpu().numpy()

                batch_size = sequence_feat.shape[0]
                for i in range(batch_size):
                    key = keys[idx]
                    residue_dict[key] = residue_feat[i]
                    sequence_dict[key] = sequence_feat[i]
                    idx += 1

        #torch.save(residue_dict, f"./hidden_layer/{save_prefix}_residue.pth")
        torch.save(sequence_dict, f"./hidden_layer/{save_prefix}_sequence.pth")
        print(f" ✅ {save_prefix} 特征提取完成")

    extract("./Dataset/AlphaGenome_train.pth", "alpha_train")
    extract("./Dataset/AlphaGenome_test.pth", "alpha_test")

def train_splicebert():
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    # =========================
    # Seed（完全一致）
    # =========================
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(42)

    dataset = SpliceBERTDataset("./Dataset/SpliceBERT_train.pth", max_len=512)
    X_all, y_all = dataset.X, dataset.y   # numpy

    num_samples, sequence_length, feature_dim = X_all.shape
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=32,
        shuffle=False
    )
    model = CNN(
        input_dim=feature_dim,
        sequence_length=sequence_length
    ).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )
    num_epochs = 100
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"[Epoch {epoch + 1:03d}] Val Loss: {val_loss:.4f}")

        # ===== 早停 =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./checkpoint/cnn_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发")
                break
    print(" SpliceBERT训练完成")



def extract_splicebert_features():
    model = CNN(input_dim=512, sequence_length=512).to(DEVICE)
    model.load_state_dict(torch.load("./checkpoint/cnn_best.pt", weights_only=True))
    model.eval()
    def extract(pth_path, save_prefix):
        dataset = SpliceBERTDataset(pth_path, max_len=512)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        keys = dataset.keys  # 关键：复用Dataset的keys（需保证SpliceBERTDataset定义了keys属性）
        residue_dict = {}
        sequence_dict = {}
        idx = 0
        with torch.no_grad():
            for x, _ in loader:  # x是Dataset输出的特征，y用不到所以用_接收
                x = x.to(DEVICE, non_blocking=True)
                feat_residue, feat_sequence = model(x, mode="both")
                feat_residue = feat_residue.cpu().numpy()
                feat_sequence = feat_sequence.cpu().numpy()

                batch_size = feat_sequence.shape[0]
                for i in range(batch_size):
                    key = keys[idx]
                    residue_dict[key] = feat_residue[i]
                    sequence_dict[key] = feat_sequence[i]
                    idx += 1

        #torch.save(residue_dict, f"./hidden_layer/{save_prefix}_residue.pth")
        torch.save(sequence_dict, f"./hidden_layer/{save_prefix}_sequence.pth")
        print(f"✔ {save_prefix} 特征提取完成")
    extract("./Dataset/SpliceBERT_train.pth", "splice_train")
    extract("./Dataset/SpliceBERT_test.pth", "splice_test")


def train_gpnmsa():
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(42)
    dataset = GPNMSADataset("./Dataset/GPN_MSA_train.pth")
    X_all, y_all = dataset.X, dataset.y   # numpy
    num_samples, sequence_length, feature_dim = X_all.shape
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=32,
        shuffle=False
    )
    model = TextCNN(
        input_dim=feature_dim,
        sequence_length=sequence_length
    ).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )
    num_epochs = 100
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"[Epoch {epoch + 1:03d}] Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./checkpoint/textcnn_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发")
                break
    print(" GPN-MSA训练完成")
    return None


def extract_gpnmsa_features():
    # 初始化模型（保持和训练时一致的参数）
    model = TextCNN(input_dim=768, sequence_length=128).to(DEVICE)
    model.load_state_dict(torch.load("./checkpoint/textcnn_best.pt", weights_only=True))
    model.eval()
    def extract(pth_path, save_prefix):
        dataset = GPNMSADataset(pth_path)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        keys = dataset.keys  # 关键：保证GPNMSADataset定义了keys属性
        residue_dict = {}
        sequence_dict = {}
        idx = 0
        with torch.no_grad():
            for x, _ in loader:  # x是Dataset输出的特征，y用不到所以用_接收
                x = x.to(DEVICE, non_blocking=True)
                feat_residue, feat_sequence = model(x, mode="both")
                feat_residue = feat_residue.cpu().numpy()
                feat_sequence = feat_sequence.cpu().numpy()

                batch_size = feat_sequence.shape[0]
                for i in range(batch_size):
                    key = keys[idx]
                    residue_dict[key] = feat_residue[i]
                    sequence_dict[key] = feat_sequence[i]
                    idx += 1
        #torch.save(residue_dict, f"./hidden_layer/{save_prefix}_residue.pth")
        torch.save(sequence_dict, f"./hidden_layer/{save_prefix}_sequence.pth")
        print(f" {save_prefix} 特征提取完成")
    extract("./Dataset/GPN_MSA_train.pth", "gpn_train")
    extract("./Dataset/GPN_MSA_test.pth", "gpn_test")



def file_exists(path):
    return os.path.exists(path)

def all_files_exist(file_list):
    return all(os.path.exists(f) for f in file_list)

def Feature_AUG_Model():
    print("AlphaGenome Stage")
    dcnn_ckpt = "./checkpoint/dcnn_best.pt"
    if file_exists(dcnn_ckpt):
        print("✔ DCNN 已存在，跳过训练")
    else:
        print("→ 开始训练 DCNN")
        train_alphagenome()
    alpha_feature_files = [
        #"./hidden_layer/alpha_train_residue.pth",
        "./hidden_layer/alpha_train_sequence.pth",
        #"./hidden_layer/alpha_test_residue.pth",
        "./hidden_layer/alpha_test_sequence.pth",
    ]
    if all_files_exist(alpha_feature_files):
        print("✔ AlphaGenome 特征已存在，跳过提取")
    else:
        print("→ 提取 AlphaGenome 特征")

        extract_alphagenome_features()

    print("\nSpliceBERT Stage")
    cnn_ckpt = "./checkpoint/cnn_best.pt"
    if file_exists(cnn_ckpt):
        print("✔ CNN 已存在，跳过训练")
    else:
        print("→ 开始训练 CNN")
        train_splicebert()
    splice_feature_files = [
        #"./hidden_layer/splice_train_residue.pth",
        "./hidden_layer/splice_train_sequence.pth",
        #"./hidden_layer/splice_test_residue.pth",
        "./hidden_layer/splice_test_sequence.pth",
    ]
    if all_files_exist(splice_feature_files):
        print("✔ SpliceBERT 特征已存在，跳过提取")
    else:
        print("→ 提取 SpliceBERT 特征")
        extract_splicebert_features()

    print("\nGPN-MSA Stage")
    textcnn_ckpt = "./checkpoint/textcnn_best.pt"
    if file_exists(textcnn_ckpt):
        print("✔ TextCNN 已存在，跳过训练")
    else:
        print("→ 开始训练 TextCNN")
        train_gpnmsa()
    gpn_feature_files = [
        #"./hidden_layer/gpn_train_residue.pth",
        "./hidden_layer/gpn_train_sequence.pth",
        #"./hidden_layer/gpn_test_residue.pth",
        "./hidden_layer/gpn_test_sequence.pth",
    ]
    if all_files_exist(gpn_feature_files):
        print("✔ GPN-MSA 特征已存在，跳过提取")
    else:
        print("→ 提取 GPN-MSA 特征")
        extract_gpnmsa_features()


def main():
    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs("./hidden_layer", exist_ok=True)
    RUN_FEATURE_PIPELINE = True
    if RUN_FEATURE_PIPELINE:
        Feature_AUG_Model()
    else:
        print("直接进入模型训练阶段")
    # =========================
    # 1. 加载特征
    # =========================
    paths = {
        "alpha_train_seq": "./hidden_layer/alpha_train_sequence.pth",
        "splice_train_seq": "./hidden_layer/splice_train_sequence.pth",
        "gpn_train_seq": "./hidden_layer/gpn_train_sequence.pth",

        "alpha_test_seq": "./hidden_layer/alpha_test_sequence.pth",
        "splice_test_seq": "./hidden_layer/splice_test_sequence.pth",
        "gpn_test_seq": "./hidden_layer/gpn_test_sequence.pth",
    }

    alpha_train = torch.load(paths["alpha_train_seq"], map_location="cpu", weights_only=False)
    splice_train = torch.load(paths["splice_train_seq"], map_location="cpu", weights_only=False)
    gpn_train = torch.load(paths["gpn_train_seq"], map_location="cpu", weights_only=False)

    alpha_test = torch.load(paths["alpha_test_seq"], map_location="cpu", weights_only=False)
    splice_test = torch.load(paths["splice_test_seq"], map_location="cpu", weights_only=False)
    gpn_test = torch.load(paths["gpn_test_seq"], map_location="cpu", weights_only=False)

    # =========================
    # 2. 构建 Dataset
    # =========================
    train_dataset = MultiModalDataset(
        alpha_train, splice_train, gpn_train
    )

    test_dataset = MultiModalDataset(
        alpha_test, splice_test, gpn_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # =========================
    # 4. 模型训练
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mode = "5"
    '''
        1: 仅使用对比损失。解耦式训练，对比学习训练门控融合，后提取融合表征训练MLP做分类
        2: 仅使用二分类损失
        3: 使用对比学习+二分类(早停)
        4: 五折交叉验证
        5: 使用平均轮次在完整训练集上训练,保存融合测试集、保存测试集分数
    '''
    if train_mode == "1":
        model = GateFusion().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        train_SupCon_decoupling(model=model, train_loader=train_loader, optimizer=optimizer, device=device, temperature=0.09)
        model.load_state_dict(
            torch.load("./checkpoint/fusion_contrast.pt", map_location=device)
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        train_feat, train_label = extract_features(model, train_loader, device)
        test_feat, test_label = extract_features(model, test_loader, device)
        train_mlp_classifier(
            train_feat, train_label,
            test_feat, test_label,
            device=device
        )
    elif train_mode == "2":      #
        #model = GateFusion_MLP_earlystop().to(device)
        model = GateFusion_MLP_5floder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        train_BCEWithLogitsLoss(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device
        )

    elif train_mode == "3":
        model = GateFusion_MLP_earlystop().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        train_SupCon_BCEWithLogitsLoss(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            temperature=1.0
        )
    elif train_mode == "4":
        model = GateFusion_MLP_5floder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        train_SupCon_BCEWithLogitsLoss_5fold(
            model=model,
            full_train_dataset=train_dataset,  # 传 Dataset，不是 DataLoader
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            temperature=1.0,
            epochs=50,
            patience=10
        )

    elif train_mode == "5":
        model = GateFusion_MLP_5floder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        train_SupCon_BCEWithLogitsLoss_fixed_epoch(
            model=model,
            train_loader=train_loader,  # 完整训练集
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            fixed_epochs=10,
            temperature=0.09
        )
    elif train_mode == "6":
        results = []
        base_model = GateFusion_MLP_5floder().to(device)
        init_state = copy.deepcopy(base_model.state_dict())

        a_list = np.arange(0, 2.5 + 0.025, 0.025)

        for a in a_list:
            # 与「多次单独运行 mode 5」一致：每个 a 的训练都从同一随机状态起跑，
            # 否则 RNG 会在循环间被顺序消耗，打乱顺序与 dropout 等与 a 混淆。
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print("\n" + "=" * 80)
            print(f"当前 a = {a:.4f}")
            print("=" * 80)

            # =====================
            # 每次重新初始化模型
            # =====================
            model = GateFusion_MLP_5floder().to(device)
            model.load_state_dict(init_state)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4
            )

            metrics = train_SupCon_BCEWithLogitsLoss_fixed_epoch_search_a(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                device=device,
                fixed_epochs=10,
                temperature=0.09,
                a=a
            )

            results.append(metrics)

        # =========================
        # 保存 CSV
        # =========================
        results_df = pd.DataFrame(results)

        save_csv = "./hidden_layer/supcon_weight_search_results.csv"

        results_df.to_csv(save_csv, index=False)

        print("\n" + "=" * 80)
        print(f"✅ 所有结果已保存:")
        print(save_csv)
        print("=" * 80)

        # =========================
        # 最优结果
        # =========================
        best_row = results_df.loc[
            results_df["AUC"].idxmax()
        ]

        print("\n===== 最优 AUC =====")
        print(best_row)










if __name__ == "__main__":
    main()