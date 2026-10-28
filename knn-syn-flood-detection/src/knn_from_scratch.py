import pandas as pd
import numpy as np
import math


def calculate(h1, h2, metric='euclidean'):
    """
    Tính khoảng cách giữa 2 vector.
    """
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)

    if metric == 'euclidean':
        return math.sqrt(np.sum((h1 - h2) ** 2))
    elif metric == 'manhattan':
        return np.sum(np.abs(h1 - h2))
    else:
        raise ValueError("metric phải là 'euclidean' hoặc 'manhattan'")


def tim_neighbors(X_train, y_train, test_row, k, metric='euclidean'):
    """
    Tìm k hàng xóm gần nhất cho 1 mẫu.
 
    """
    if metric == 'euclidean':
        distances = np.sqrt(np.sum((X_train - test_row) ** 2, axis=1))
    elif metric == 'manhattan':
        distances = np.sum(np.abs(X_train - test_row), axis=1)
    else:
        raise ValueError("metric phải là 'euclidean' hoặc 'manhattan'")

    k = min(k, len(X_train))
    chi_so_gan_nhat = np.argpartition(distances, k - 1)[:k]
    chi_so_gan_nhat = chi_so_gan_nhat[np.argsort(distances[chi_so_gan_nhat])]

    neighbors = [(int(y_train[i]), float(distances[i])) for i in chi_so_gan_nhat]
    return neighbors


def predict(X_train, y_train, test_row, k, metric='euclidean', voting='uniform'):
    """
    Dự đoán cho 1 mẫu.
    """
    neighbors = tim_neighbors(X_train, y_train, test_row, k, metric=metric)

    if voting == 'distance':
        diem = {}
        for label, dist in neighbors:
            w = 1.0 / (dist + 1e-12)
            diem[label] = diem.get(label, 0.0) + w
        return int(max(diem, key=diem.get))

    nhan = [item[0] for item in neighbors]
    unique_labels, counts = np.unique(nhan, return_counts=True)
    max_count = np.max(counts)
    cac_lop_tie = unique_labels[counts == max_count]

    if len(cac_lop_tie) == 1:
        return int(cac_lop_tie[0])

    tong_khoang_cach_theo_lop = {}
    for lop in cac_lop_tie:
        tong_khoang_cach_theo_lop[lop] = sum(
            dist for label, dist in neighbors if label == lop
        )

    return int(min(tong_khoang_cach_theo_lop, key=tong_khoang_cach_theo_lop.get))


def chuan_hoa(X_train, X_test):
    """
    Chuẩn hóa dữ liệu theo mean/std của tập train.
    
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled


def tai_du_lieu(train_path='syn_train.csv', test_path='syn_test.csv'):
    print("Loading data...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['Label'] = train_df['Label'].apply(lambda x: 1 if x == 'Syn' else 0)
    test_df['Label'] = test_df['Label'].apply(lambda x: 1 if x == 'Syn' else 0)

    X_train = train_df.drop('Label', axis=1).values.astype(np.float32)
    y_train = train_df['Label'].values.astype(np.int32)

    X_test = test_df.drop('Label', axis=1).values.astype(np.float32)
    y_test = test_df['Label'].values.astype(np.int32)

    return train_df, test_df, X_train, y_train, X_test, y_test


def du_doan_toan_bo(X_train, y_train, X_test, k, metric='euclidean', voting='uniform', buoc_hien_thi=100, hien_thi=True):
    """
    Dự đoán toàn bộ tập test.
    """
    tong_so_test = len(X_test)
    y_pred = np.empty(tong_so_test, dtype=np.int32)

    if hien_thi:
        print(f"Predicting {tong_so_test} network flows")

    for i in range(tong_so_test):
        y_pred[i] = predict(X_train, y_train, X_test[i], k, metric=metric, voting=voting)

        if hien_thi and ((i + 1) % buoc_hien_thi == 0 or (i + 1) == tong_so_test):
            print(f"  Processed {i+1}/{tong_so_test} flows")

    return y_pred


def tinh_ma_tran_nham_lan(y_test, y_pred):
    TP = int(np.sum((y_test == 1) & (y_pred == 1)))
    TN = int(np.sum((y_test == 0) & (y_pred == 0)))
    FP = int(np.sum((y_test == 0) & (y_pred == 1)))
    FN = int(np.sum((y_test == 1) & (y_pred == 0)))
    return TP, TN, FP, FN


def tinh_chi_so(y_test, y_pred):
    TP, TN, FP, FN = tinh_ma_tran_nham_lan(y_test, y_pred)

    tong_so = len(y_test)
    accuracy = (TP + TN) / tong_so if tong_so > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    far = FP / (TN + FP) if (TN + FP) > 0 else 0
    fnr = FN / (TP + FN) if (TP + FN) > 0 else 0

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "far": far,
        "fnr": fnr
    }


def in_ket_qua(chi_so, tong_so_test):
    print(f"\n========= KẾT QUẢ THU ĐƯỢC TRÊN {tong_so_test} FLOWS =========")
    print(f"Overall Accuracy: {chi_so['accuracy'] * 100:.2f}%")
    print(f"Precision       : {chi_so['precision'] * 100:.2f}%")
    print(f"Recall          : {chi_so['recall'] * 100:.2f}%")
    print(f"F1-Score        : {chi_so['f1_score'] * 100:.2f}%")
    print(f"False Alarm Rate: {chi_so['far'] * 100:.2f}%")
    print(f"False Neg Rate  : {chi_so['fnr'] * 100:.2f}%\n")

    print(f" True Positive  : {chi_so['TP']}")
    print(f" True Negative  : {chi_so['TN']}")
    print(f" False Positive : {chi_so['FP']}")
    print(f" False Negative : {chi_so['FN']}")


def luu_ket_qua(test_df_goc, y_test, y_pred, output_file="ket_qua_du_doan.csv"):
    df_ket_qua = test_df_goc.copy()
    df_ket_qua["y_thuc_te"] = y_test
    df_ket_qua["y_du_doan"] = y_pred
    df_ket_qua["ket_qua"] = np.where(y_test == y_pred, "Đúng", "Sai")
    df_ket_qua.to_csv(output_file, index=False)
    print(f"\nĐã lưu kết quả dự đoán vào file: {output_file}")


def luu_mau_du_doan_sai(test_df_goc, y_test, y_pred, output_file="mau_du_doan_sai.csv"):
    df = test_df_goc.copy()
    df["y_thuc_te"] = y_test
    df["y_du_doan"] = y_pred
    df_sai = df[y_test != y_pred].copy()
    df_sai.to_csv(output_file, index=False)
    print(f"Đã lưu các mẫu dự đoán sai vào file: {output_file}")


def demo_mot_mau(X_train, y_train, X_test, y_test, index=0, k=5, metric='euclidean', voting='uniform'):
    test_row = X_test[index]
    neighbors = tim_neighbors(X_train, y_train, test_row, k, metric=metric)
    du_doan = predict(X_train, y_train, test_row, k, metric=metric, voting=voting)
    thuc_te = y_test[index]

    print("\n========= DEMO 1 FLOW =========")
    print(f"Flow index      : {index}")
    print(f"Nhãn thực tế    : {thuc_te} ({'Syn' if thuc_te == 1 else 'Normal'})")
    print(f"Nhãn dự đoán    : {du_doan} ({'Syn' if du_doan == 1 else 'Normal'})")
    print("Hàng xóm gần nhất:")
    for i, (label, dist) in enumerate(neighbors, start=1):
        print(f"  Neighbor {i}: label={label}, distance={dist:.6f}")


def tao_folds(n, so_fold=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return np.array_split(indices, so_fold)


def tinh_ma_tran_khoang_cach(X_a, X_b, metric='euclidean'):
    """
    Tính ma trận khoảng cách giữa mọi mẫu trong X_a và X_b.
    Kết quả shape: (len(X_a), len(X_b))
    """
    if metric == 'euclidean':
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        a2 = np.sum(X_a ** 2, axis=1, keepdims=True)
        b2 = np.sum(X_b ** 2, axis=1)
        dist2 = a2 + b2 - 2 * (X_a @ X_b.T)
        dist2 = np.maximum(dist2, 0.0)
        return np.sqrt(dist2)
    elif metric == 'manhattan':
       
        return np.sum(np.abs(X_a[:, None, :] - X_b[None, :, :]), axis=2)
    else:
        raise ValueError("metric phải là 'euclidean' hoặc 'manhattan'")


def du_doan_tu_ma_tran_khoang_cach(dist_matrix, y_train, k, voting='uniform'):
    """
    Dự đoán cho nhiều mẫu cùng lúc từ ma trận khoảng cách đã tính sẵn.
    """
    n_test = dist_matrix.shape[0]
    y_pred = np.empty(n_test, dtype=np.int32)

    # Lấy k láng giềng gần nhất cho toàn bộ test
    idx_k = np.argpartition(dist_matrix, k - 1, axis=1)[:, :k]

    # Sắp xếp lại theo khoảng cách trong nhóm k láng giềng
    row_idx = np.arange(n_test)[:, None]
    dist_k = dist_matrix[row_idx, idx_k]
    order = np.argsort(dist_k, axis=1)
    idx_k = idx_k[row_idx, order]
    dist_k = dist_k[row_idx, order]

    labels_k = y_train[idx_k]

    if voting == 'distance':
        weights = 1.0 / (dist_k + 1e-12)
        diem_1 = np.sum(weights * (labels_k == 1), axis=1)
        diem_0 = np.sum(weights * (labels_k == 0), axis=1)
        y_pred = (diem_1 > diem_0).astype(np.int32)

        tie_mask = diem_1 == diem_0
        if np.any(tie_mask):
            tong_dist_1 = np.sum(np.where(labels_k[tie_mask] == 1, dist_k[tie_mask], 0), axis=1)
            tong_dist_0 = np.sum(np.where(labels_k[tie_mask] == 0, dist_k[tie_mask], 0), axis=1)
            y_pred[tie_mask] = (tong_dist_1 < tong_dist_0).astype(np.int32)

        return y_pred

    # uniform voting
    so_phieu_1 = np.sum(labels_k == 1, axis=1)
    so_phieu_0 = k - so_phieu_1

    y_pred = (so_phieu_1 > so_phieu_0).astype(np.int32)

    tie_mask = so_phieu_1 == so_phieu_0
    if np.any(tie_mask):
        tong_dist_1 = np.sum(np.where(labels_k[tie_mask] == 1, dist_k[tie_mask], 0), axis=1)
        tong_dist_0 = np.sum(np.where(labels_k[tie_mask] == 0, dist_k[tie_mask], 0), axis=1)
        y_pred[tie_mask] = (tong_dist_1 < tong_dist_0).astype(np.int32)

    return y_pred


def chon_k_tot_nhat(X_train, y_train, ds_k=(1, 3, 5, 7, 9, 11), so_fold=5, metric='euclidean', voting='uniform'):
    """
    Chọn k tốt nhất bằng cross-validation, tối ưu bằng cách:
    mỗi fold chỉ tính khoảng cách 1 lần, rồi dùng lại cho mọi k.
    """
    folds = tao_folds(len(X_train), so_fold=so_fold, seed=42)
    ket_qua = []

    for k in ds_k:
        pass

    diem_theo_k = {k: [] for k in ds_k}

    for i in range(so_fold):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(so_fold) if j != i])

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        print(f"CV Fold {i+1}/{so_fold}: train={len(X_tr)}, val={len(X_val)}")

        # Chỉ tính 1 lần cho fold này
        dist_matrix = tinh_ma_tran_khoang_cach(X_val, X_tr, metric=metric)

        for k in ds_k:
            y_val_pred = du_doan_tu_ma_tran_khoang_cach(
                dist_matrix, y_tr, k, voting=voting
            )
            chi_so = tinh_chi_so(y_val, y_val_pred)
            diem_theo_k[k].append(chi_so["f1_score"])

    for k in ds_k:
        f1_tb = float(np.mean(diem_theo_k[k]))
        ket_qua.append((k, f1_tb))

    ket_qua = sorted(ket_qua, key=lambda x: x[1], reverse=True)
    return ket_qua[0][0], ket_qua


def main():
    train_df, test_df, X_train, y_train, X_test, y_test = tai_du_lieu()

    X_train_scaled, X_test_scaled = chuan_hoa(X_train, X_test)

    metric = 'euclidean'
    voting = 'distance'

    ds_k = [1, 3, 5, 7, 9, 11]
    k_tot_nhat, bang_cv = chon_k_tot_nhat(
        X_train_scaled, y_train,
        ds_k=ds_k,
        so_fold=5,
        metric=metric,
        voting=voting
    )

    print("\n========= K-FOLD CV =========")
    for k, f1_tb in bang_cv:
        print(f"k={k:<2} | F1 trung bình = {f1_tb * 100:.4f}%")
    print(f"--> Chọn k tốt nhất trên tập train: {k_tot_nhat}")

    y_pred = du_doan_toan_bo(
        X_train_scaled, y_train, X_test_scaled,
        k_tot_nhat,
        metric=metric,
        voting=voting,
        hien_thi=True
    )

    chi_so = tinh_chi_so(y_test, y_pred)
    in_ket_qua(chi_so, len(X_test_scaled))

    demo_mot_mau(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        index=0,
        k=k_tot_nhat,
        metric=metric,
        voting=voting
    )

    luu_ket_qua(test_df, y_test, y_pred, output_file="ket_qua_du_doan.csv")
    luu_mau_du_doan_sai(test_df, y_test, y_pred, output_file="mau_du_doan_sai.csv")


if __name__ == "__main__":
    main()