from src.utils.utils import fix_seed, train
from src.utils.logger import local_logger, LOG_PATH
from src.models.params import ModelParams, ModuleParams

import hydra
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path
import torch

import warnings

# turn off the UserWarnings because lots of them are talking about 
# library function refactoring, last or future deprecations
warnings.filterwarnings('ignore', category=UserWarning)

@fix_seed
@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    time_log_marker = datetime.today().strftime("%d_%m_%Y_%H_%M")

    # Определяем пространство поиска гиперпараметров

    # Параметры для обучения с AngularMarginSoftmax
    # Для AngularMarginSoftmax часто требуются специфические значения margin и scale.
    # Типичные значения для ArcFace: m=0.5, s=64.
    params_am = ModuleParams(
        dataset_dir=Path("./data/train"),
        use_cache=False, # Можно установить в True, если датасет полностью помещается в RAM.
        checkpoints_dir=Path("./checkpoints"),
        model_params=ModelParams(
            emb_size=cfg.module.model.emb_size
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=cfg.train.num_workers,
        n_epochs=cfg.train.n_epochs, # Увеличиваем количество эпох, т.к. AM-Softmax сходится медленнее
        log_dir=Path(f"{LOG_PATH}/angular_margin_{time_log_marker}"),
        loss_function=cfg.module.loss_fn, # "cross_entropy" or "angular_margin"
        angular_margin=cfg.module.angular_margine,
        angular_scale=cfg.module.angular_scale,
        validation_dir=Path("./data/dev"),
        validation_frequency=1,
        learning_rate=cfg.module.learning_rate,
        batch_size=cfg.train.batch_size,
        save_best_checkpoint=True
    )

    local_logger.info(f"Starting training: tensorboard time log marker: {time_log_marker}")
    local_logger.info(f"Current model params: {params_am}")
    _ = train(params_am)
    local_logger.info("Training completed")

if __name__ == "__main__":
    main()