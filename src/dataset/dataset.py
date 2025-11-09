import torch
from pathlib import Path
import torchaudio

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