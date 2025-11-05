import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
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
    model.eval() # Переводим модель в режим оценки
    
    embeddings_list: List[torch.Tensor] = []
    labels_list: List[int] = []

    with torch.no_grad(): # Отключаем расчет градиентов
        for _, wavs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            # В `Conformer` `forward` возвращает (emb, scores). Нам нужен `emb` для EER.
            emb, _ = model.forward(wavs.to(device))
            embeddings_list.append(emb.cpu()) # Перемещаем эмбеддинги на CPU
            labels_list.extend(labels.tolist())

    all_embeddings = torch.cat(embeddings_list, dim=0).numpy()
    all_labels = torch.tensor(labels_list).numpy()

    # Создаем пары для сравнения
    # Для простоты, сгенерируем все возможные пары (можно оптимизировать для больших датасетов)
    # или генерировать фиксированное количество положительных/отрицательных пар.
    # Здесь мы будем сравнивать каждый эмбеддинг с каждым.

    scores = [] # Список для косинусного сходства
    actual_is_same = [] # Список, указывающий, являются ли пары от одного человека (1) или разных (0)

    # Нормализуем все эмбеддинги перед сравнением
    all_embeddings = all_embeddings / (
        torch.norm(torch.from_numpy(all_embeddings), dim=1, keepdim=True).numpy() + 1e-9
    )

    num_samples = len(all_labels)
    for i in range(num_samples):
        for j in range(i + 1, num_samples): # Избегаем дубликатов и сравнения с самим собой
            # Косинусное сходство
            similarity = torch.dot(
                torch.from_numpy(all_embeddings[i]), torch.from_numpy(all_embeddings[j])
            ).item()
            scores.append(similarity)
            actual_is_same.append(1 if all_labels[i] == all_labels[j] else 0)

    scores = torch.tensor(scores)
    actual_is_same = torch.tensor(actual_is_same)

    # Вычисление FAR, FRR и EER
    # roc_curve возвращает FPR (False Positive Rate, то же что и FAR) и TPR (True Positive Rate)
    fpr, tpr, thresholds = roc_curve(actual_is_same, scores)
    
    # FRR = 1 - TPR
    frr = 1 - tpr

    # EER - это точка, где FAR (fpr) == FRR (1 - tpr)
    # scipy.optimize.brentq находит корень функции f(x) = 0 в заданном интервале [a, b].
    # Мы ищем корень функции f(threshold) = fpr(threshold) - frr(threshold)
    try:
        eer = brentq(lambda x: 1.0 - tpr[torch.argmin(torch.abs(torch.from_numpy(thresholds - x)))] - fpr[torch.argmin(torch.abs(torch.from_numpy(thresholds - x)))],
                     min(scores), max(scores))
        
        # Получаем соответствующий FAR/FRR в точке EER
        eer_threshold_idx = torch.argmin(torch.abs(torch.from_numpy(thresholds - eer)))
        eer_far = fpr[eer_threshold_idx]
        eer_frr = frr[eer_threshold_idx]
        eer = (eer_far + eer_frr) / 2 # EER это среднее FAR и FRR в точке равенства

    except ValueError:
        # В случае, если brentq не может найти корень (например, если нет пересечения),
        # это может произойти, если пороги слишком ограничены или данных недостаточно.
        # В таких случаях можно попробовать более простую интерполяцию или просто вернуть 1.0.
        # Для стабильности и предотвращения ошибок, можно вернуть наилучшее приближение или 1.0.
        print("Warning: Could not find EER using brentq. Approximating EER.")
        min_abs_diff = float('inf')
        eer_val = 1.0
        for i in range(len(fpr)):
            diff = abs(fpr[i] - frr[i])
            if diff < min_abs_diff:
                min_abs_diff = diff
                eer_val = (fpr[i] + frr[i]) / 2.0
        eer = eer_val
        
    model.train() # Возвращаем модель в режим обучения
    return eer


def main(conf: ModuleParams) -> None:

    conf.log_dir.mkdir(exist_ok=True, parents=True)
    conf.checkpoints_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=conf.log_dir)

    dataset = SpeakerDataset(dataset_dir=conf.dataset_dir, use_cache=conf.use_cache)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=True
    )

    val_dataloader = None
    if conf.validation_dir and conf.validation_dir.exists():
        val_dataset = SpeakerDataset(
            dataset_dir=conf.validation_dir, use_cache=conf.use_cache
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=conf.batch_size,
            num_workers=conf.num_workers,
            shuffle=False,
        )

    n_classes = len(set(dataset.labels))

    model_params = conf.model_params
    model_params = replace(model_params, n_classes=n_classes)

    model = Conformer(model_params).to(conf.device)

    if conf.loss_function == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif conf.loss_function == "angular_margin":
        # Убедитесь, что angular_margin и angular_scale заданы
        if conf.angular_margin is None or conf.angular_scale is None:
            raise ValueError(
                "angular_margin and angular_scale must be provided for 'angular_margin' loss function."
            )
        criterion = AngularMarginSoftmax(
            embedding_dim=model_params.emb_size,
            num_classes=n_classes,
            margin=conf.angular_margin,
            scale=conf.angular_scale,
        ).to(conf.device)
    else:
        raise ValueError(f"Invalid loss function: {conf.loss_function}")

    optim = torch.optim.Adam(params=model.parameters(), lr=conf.learning_rate)

    pbar = tqdm(range(conf.n_epochs), position=0, leave=True)

    global_step = 0

    best_eer = float('inf') # Для сохранения лучшей модели по EER

    for epoch in pbar:
        model.train() # Переключаем модель в режим обучения
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for batch in train_dataloader:
            _, wavs, labels = batch

            # _, scores = model.forward(wavs.to(conf.device))
            if conf.loss_function == "cross_entropy":
                _, scores = model.forward(wavs.to(conf.device))
                loss = criterion(scores, labels.to(conf.device))
                predictions = torch.argmax(scores, dim=1)
            elif conf.loss_function == "angular_margin":
                embeddings, _ = model.forward(wavs.to(conf.device)) # Получаем эмбеддинги
                loss = criterion(embeddings, labels.to(conf.device))
                # Для AngularMarginSoftmax, предсказания делаются с помощью метода predict()
                predictions = criterion.predict(embeddings)
            else:
                raise ValueError("Unexpected loss function during training loop.")

            optim.zero_grad()

            # loss = criterion(scores, labels.to(conf.device))

            loss.backward()
            optim.step()

            # predictions = torch.argmax(scores, dim=1)

            correct = (predictions == labels.to(conf.device)).sum().item()
            epoch_correct += correct
            epoch_total += labels.size(0)

            epoch_losses.append(loss.item())

            writer.add_scalar("Loss/Batch", loss.item(), global_step)
            writer.add_scalar("Accuracy/Batch", correct / labels.size(0), global_step)
            writer.add_scalar("Learning_Rate", optim.param_groups[0]["lr"], global_step)

            global_step += 1

            pbar.set_postfix({"batch_loss": f"{loss.item():.2f}", "batch_acc": f"{correct / labels.size(0):.2f}"})

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_accuracy = epoch_correct / epoch_total

        writer.add_scalar("Loss/Epoch", avg_epoch_loss, epoch)
        writer.add_scalar("Accuracy/Epoch", epoch_accuracy, epoch)

        # Выводим точность тренировки
        print(f"Epoch {epoch + 1}: Avg Train Loss = {avg_epoch_loss:.4f}, Train Accuracy = {epoch_accuracy:.4f}")

        if val_dataloader and (epoch + 1) % conf.validation_frequency == 0:
            print(f"\nRunning validation evaluation at epoch {epoch + 1}...")
            eer = evaluate(model, val_dataloader, conf.device)
            writer.add_scalar("Validation/EER", eer, epoch)
            print(f"Validation EER = {eer:.4f}")

            # Сохраняем модель, если EER улучшился
            if eer < best_eer:
                best_eer = eer
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "best_eer": best_eer,
                        "loss_function": conf.loss_function,
                        "angular_margin": (
                            conf.angular_margin
                            if conf.loss_function == "angular_margin"
                            else None
                        ),
                        "angular_scale": (
                            conf.angular_scale
                            if conf.loss_function == "angular_margin"
                            else None
                        ),
                    },
                    conf.checkpoints_dir / f"best_model_eer_{best_eer:.4f}.ckpt",
                )
                print(f"Saved best model with EER: {best_eer:.4f}")

        # Сохраняем чекпоинт в конце каждой эпохи (или по другой логике)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": avg_epoch_loss,
                "accuracy": epoch_accuracy,
                "loss_function": conf.loss_function,
                "angular_margin": (
                    conf.angular_margin
                    if conf.loss_function == "angular_margin"
                    else None
                ),
                "angular_scale": (
                    conf.angular_scale
                    if conf.loss_function == "angular_margin"
                    else None
                ),
            },
            conf.checkpoints_dir / f"epoch_{epoch + 1}.ckpt",
        )

    writer.close()


if __name__ == "__main__":
    # Параметры для обучения с AngularMarginSoftmax
    # Для AngularMarginSoftmax часто требуются специфические значения margin и scale.
    # Типичные значения для ArcFace: m=0.5, s=64.
    params_am = ModuleParams(
        dataset_dir=Path("./data/train"),
        use_cache=False, # Можно установить в True, если датасет полностью помещается в RAM.
        checkpoints_dir=Path("./checkpoints"),
        model_params=ModelParams(
            emb_size=128 # Увеличиваем размер эмбеддинга для лучшего разделения в угловом пространстве
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=4 if torch.cuda.is_available() else 1,
        n_epochs=50, # Увеличиваем количество эпох, т.к. AM-Softmax сходится медленнее
        log_dir=Path("./logs/angular_margin_hw_test"),
        loss_function="angular_margin",
        angular_margin=0.5,
        angular_scale=64.0,
        validation_dir=Path("./data/dev"),
        validation_frequency=1,
        learning_rate=1e-3,
        batch_size=4
    )
    print("Starting training with AngularMarginSoftmax...")
    main(params_am)