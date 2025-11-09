import torch
from typing import List
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
import numpy as np

def evaluate(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str
) -> float:
    model.eval()
    
    embeddings_list: List[torch.Tensor] = []
    labels_list: List[int] = []

    with torch.no_grad():
        for _, wavs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            emb, _ = model.forward(wavs.to(device))
            embeddings_list.append(emb.cpu())
            labels_list.extend(labels.tolist())

    all_embeddings = torch.cat(embeddings_list, dim=0)
    all_labels = torch.tensor(labels_list)

    # Нормализация эмбеддингов
    all_embeddings = all_embeddings / (torch.norm(all_embeddings, dim=1, keepdim=True) + 1e-9)

    # ОПТИМИЗАЦИЯ: Векторизованное вычисление косинусного сходства
    # Вычисляем матрицу сходства размером (N, N) за одну операцию
    similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T)
    
    # Создаем маску для верхнего треугольника (исключая диагональ)
    num_samples = len(all_labels)
    triu_indices = torch.triu_indices(num_samples, num_samples, offset=1)
    
    # Извлекаем только верхний треугольник матрицы сходства
    scores = similarity_matrix[triu_indices[0], triu_indices[1]]
    
    # Векторизованное создание меток: сравниваем метки для всех пар
    labels_i = all_labels[triu_indices[0]]
    labels_j = all_labels[triu_indices[1]]
    actual_is_same = (labels_i == labels_j).long()

    # Преобразуем в numpy для sklearn
    scores_np = scores.numpy()
    actual_is_same_np = actual_is_same.numpy()

    # Вычисление FAR, FRR и EER
    fpr, tpr, thresholds = roc_curve(actual_is_same_np, scores_np)
    frr = 1 - tpr

    # Вычисление EER
    try:
        eer = brentq(
            lambda x: 1.0 - tpr[np.argmin(np.abs(thresholds - x))] - fpr[np.argmin(np.abs(thresholds - x))],
            min(scores_np), max(scores_np)
        )
        
        eer_threshold_idx = np.argmin(np.abs(thresholds - eer))
        eer_far = fpr[eer_threshold_idx]
        eer_frr = frr[eer_threshold_idx]
        eer = (eer_far + eer_frr) / 2

    except ValueError:
        print("Warning: Could not find EER using brentq. Approximating EER.")
        min_abs_diff = float('inf')
        eer_val = 1.0
        for i in range(len(fpr)):
            diff = abs(fpr[i] - frr[i])
            if diff < min_abs_diff:
                min_abs_diff = diff
                eer_val = (fpr[i] + frr[i]) / 2.0
        eer = eer_val
        
    model.train()
    return eer