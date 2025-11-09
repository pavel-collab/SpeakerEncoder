import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from tqdm.auto import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
import numpy as np
from typing import List

class SpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path, use_cache: bool = True):
        super().__init__()

        self.wav_files = list(dataset_dir.rglob("*.wav")) # rglob используется для рекурсивного поиска, возващает итератор (генератор)
        self.labels = []
        self.cached_data: dict[int, tuple] = {}
        self.use_cache = use_cache

        class2idx = {}
        last_class_idx = -1
        for path in self.wav_files:
            class_name = path.parent.stem # stem возвращает имя файла без расширения

            if class_name not in class2idx:
                last_class_idx += 1
                class2idx[class_name] = last_class_idx
            self.labels.append(class2idx[class_name])

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        if self.use_cache:
            if idx not in self.cached_data:
                wav, _ = torchaudio.load(self.wav_files[idx])
                self.cached_data[idx] = (idx, wav[0], self.labels[idx])
            return self.cached_data[idx]
        else:
            wav, _ = torchaudio.load(self.wav_files[idx])
            return (idx, wav[0], self.labels[idx])


class StackingSubsampling(torch.nn.Module):
    def __init__(self, stride, feat_in, feat_out):
        super().__init__()
        self.stride = stride
        self.out = torch.nn.Linear(stride * feat_in, feat_out)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = features.size()
        pad_size = (self.stride - (t % self.stride)) % self.stride
        features = torch.nn.functional.pad(features, (0, 0, 0, pad_size))
        _, t, _ = features.size()
        features = torch.reshape(features, (b, t // self.stride, d * self.stride))
        out_features = self.out(features)
        out_length = torch.div(
            features_length + pad_size, self.stride, rounding_mode="floor"
        )
        return out_features, out_length


class StatisticsPooling(torch.nn.Module):
    @staticmethod
    def get_length_mask(length):
        """
        length: B
        """
        max_len = length.max().long().item()

        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        return mask.to(length.dtype)

    def forward(self, encoded, encoded_len):
        """
        encoded: B x T x D
        encoded_len: B
        return: B x 2D
        """

        mask = self.get_length_mask(encoded_len).unsqueeze(2)  # B x T x 1

        total = encoded_len.unsqueeze(1)

        avg = (encoded * mask).sum(dim=1) / total

        std = torch.sqrt(
            (mask * (encoded - avg.unsqueeze(dim=1)) ** 2).sum(dim=1) / total
        )

        return torch.cat((avg, std), dim=1)


# Фактически, этот класс выступает в качестве функции потерь
# он выдает некоторую ошибку по заданным эмбеддингами и целевым меткам
class AngularMarginSoftmax(torch.nn.Module):
    """
    Angular Margin Softmax Loss (ArcFace variant)
    https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self, embedding_dim: int, num_classes: int, margin: float, scale: float
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Веса для каждого класса (центроиды)
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        torch.nn.init.xavier_uniform_(self.weight) # Инициализация весов

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin) # Пороговое значение для предотвращения инверсии порядка
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: B x D (batch_size x embedding_dim)
        labels: B (batch_size)
        return: scalar tensor (loss)
        """
        # Нормализация эмбеддингов до единичной длины
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # Нормализация весов классов до единичной длины
        norm_weight = torch.nn.functional.normalize(self.weight, p=2, dim=1)

        # Вычисление косинусного сходства между эмбеддингами и весами классов
        # (B x D) @ (D x C) -> B x C
        cosine = torch.nn.functional.linear(norm_embeddings, norm_weight)
        
        # Получаем косинус для правильных классов
        # detach() нужен, чтобы не вычислять градиенты для этих тензоров, так как они используются для создания маски,
        # а не для обновления весов.
        # В данном случае, у нас реализован подход ArcFace
        # По-сути, мы учим модель тому, что эмбеддинг должен находится в пределах некоторого марджина
        # от угла заданного класса
        sine = torch.sqrt(1.0 - torch.pow(cosine.detach(), 2))
        phi = cosine * self.cos_m - sine * self.sin_m # cos(theta + m)

        # Если cos(theta) < cos(pi - m), то theta + m > pi, что может привести к инверсии порядка.
        # В этом случае вместо cos(theta + m) используем cos(theta) - sin(pi - m) * m
        # (Это часть реализации ArcFace для стабильности, смотрите оригинальную статью)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Создаем one-hot вектор для меток
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Применяем модификацию только к правильным классам
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        # Используем CrossEntropyLoss, которая включает LogSoftmax
        # reduce=True по умолчанию, усредняет по батчу
        return torch.nn.functional.cross_entropy(output, labels)

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: B x D
        return: B (predicted class indices)
        """
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        norm_weight = torch.nn.functional.normalize(self.weight, p=2, dim=1)
        
        # Вычисление косинусного сходства
        cosine = torch.nn.functional.linear(norm_embeddings, norm_weight)
        
        # Прогнозируем класс с наибольшим сходством
        return torch.argmax(cosine, dim=1)


class SpecScaler(torch.nn.Module):
    def forward(self, spectrogram):
        return torch.log(spectrogram.clamp_(1e-9, 1e9))


class Conformer(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=conf.sample_rate,
                n_fft=conf.n_fft,
                win_length=conf.win_length,
                hop_length=conf.hop_length,
                n_mels=conf.n_mels,
            ),
            SpecScaler(),
        )
        self.subsampling = StackingSubsampling(conf.stride, conf.feat_in, conf.d_model)
        self.backbone = torchaudio.models.Conformer(
            input_dim=conf.d_model,
            num_heads=conf.n_heads,
            ffn_dim=conf.d_model * conf.ff_exp_factor,
            num_layers=conf.n_layers,
            depthwise_conv_kernel_size=conf.kernel_size,
            dropout=conf.dropout,
        )
        self.pooler = StatisticsPooling()
        self.extractor = torch.nn.Sequential(
            torch.nn.Linear(2 * conf.d_model, conf.d_model),
            torch.nn.ELU(),
            torch.nn.Linear(conf.d_model, conf.emb_size),
            torch.nn.ELU(),
        )
        # Обратите внимание: слой `self.proj` не нужен, если используется AngularMarginSoftmax,
        # так как она уже включает в себя логику классификации.
        # Однако, если вы хотите использовать `forward` Conformer'а для получения scores
        # в случае `CrossEntropyLoss`, его можно оставить.
        # Для AngularMarginSoftmax `scores` будут использоваться только в функции `criterion`.
        self.proj = torch.nn.Sequential(torch.nn.Linear(conf.emb_size, conf.n_classes))


    def forward(self, wavs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        features = self.transform(wavs)

        # Длины фич после MelSpectrogram, которые имеют форму B x D x T. T - это количество фреймов.
        # Каждый фрейм имеет размерность D (n_mels).
        # Длина последовательности для каждого элемента батча - это количество фреймов.
        features_length = (
            torch.ones(features.shape[0], device=features.device) * features.shape[2]
        ).to(torch.long)

        features = features.transpose(1, 2)  # B x D x T -> B x T x D
        features, features_length = self.subsampling(features, features_length)
        encoded, encoded_len = self.backbone(features, features_length)
        emb = self.pooler(encoded, encoded_len)
        emb = self.extractor(emb)
        scores = self.proj(emb) # Этот слой используется для CrossEntropyLoss
                                # Для AngularMarginSoftmax он не будет использоваться напрямую для потерь,
                                # но может быть полезен для отладки или других целей.
        return emb, scores


@dataclass
class ModelParams:
    stride: int = 8
    feat_in: int = 64
    d_model: int = 32
    n_heads: int = 4
    ff_exp_factor: int = 2
    n_layers: int = 2
    kernel_size: int = 5
    dropout: float = 0.0
    emb_size: int = 16 # Размерность эмбеддинга
    n_classes: int = 377 # Будет заменено на реальное количество классов
    sample_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64


@dataclass
class ModuleParams:
    dataset_dir: Path
    checkpoints_dir: Path
    log_dir: Path
    model_params: ModelParams
    angular_margin: float | None = None
    angular_scale: float | None = None
    use_cache: bool = True
    device: str = "cuda"
    n_epochs: int = 100
    batch_size: int = 16
    num_workers: int = 3
    learning_rate: float = 1e-2
    loss_function: str = "cross_entropy"  # "cross_entropy" or "angular_margin"
    validation_dir: Path | None = None
    validation_frequency: int = 5


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