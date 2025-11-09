import math
import torch
import torchaudio

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