import sys
import os.path
#! need to import code from modules in parent directory
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from src.utils.utils import fix_seed, objective
from src.utils.logger import local_logger
import hydra
from omegaconf import DictConfig

import optuna
from optuna.visualization import *

import warnings

# turn off the UserWarnings because lots of them are talking about 
# library function refactoring, last or future deprecations
warnings.filterwarnings('ignore', category=UserWarning)

@fix_seed
@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    os.makedirs(f"{cfg.optuna.image_path}", exist_ok=True)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, cfg.train.n_epochs), n_trials=cfg.optuna.n_trials)

    try:
        # Сохраняем графики в папку results/
        fig_hist = plot_optimization_history(study)
        fig_hist.write_image(f"{cfg.optuna.image_path}/history.png")

        fig_contours = plot_contour(study)
        fig_contours.write_image(f"{cfg.optuna.image_path}/contour.png")

        fig_importance = plot_param_importances(study)
        fig_importance.write_image(f"{cfg.optuna.image_path}/param_importance.png")

        fig_parallel = plot_parallel_coordinate(study)
        fig_parallel.write_image(f"{cfg.optuna.image_path}/parallel_coordinates.png")
    except Exception as ex:
        local_logger.error(f"Can't save trial plots because of error: {ex}")
    else:
        local_logger.info(f"Все графики успешно сохранены")
    finally:
        local_logger.info(f"\nЛучшая комбинация параметров:\n {study.best_params}")
        local_logger.info(f"\nЛучшая точность: {study.best_value:.4f}")

if __name__ == "__main__":
    main()