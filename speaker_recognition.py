from utils import fix_seed, objective
from logger import local_logger

import optuna
from optuna.visualization import *
import os

import warnings

# turn off the UserWarnings because lots of them are talking about 
# library function refactoring, last or future deprecations
warnings.filterwarnings('ignore', category=UserWarning)

N_TRIALS = 1 #TODO: посмотреть, что будет, если менять этот параметр
PATH_TO_IMAGES = "./result_images"

@fix_seed
def main() -> None:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    try:
        # Сохраняем графики в папку results/
        fig_hist = plot_optimization_history(study)
        fig_hist.write_image(f"{PATH_TO_IMAGES}/history.png")

        fig_contours = plot_contour(study)
        fig_contours.write_image(f"{PATH_TO_IMAGES}/contour.png")

        fig_importance = plot_param_importances(study)
        fig_importance.write_image(f"{PATH_TO_IMAGES}/param_importance.png")

        fig_parallel = plot_parallel_coordinate(study)
        fig_parallel.write_image(f"{PATH_TO_IMAGES}/parallel_coordinates.png")
    except Exception as ex:
        local_logger.error(f"Can't save trial plots because of error: {ex}")
    else:
        local_logger.info(f"Все графики успешно сохранены")
    finally:
        local_logger.info(f"\nЛучшая комбинация параметров:\n {study.best_params}")
        local_logger.info(f"\nЛучшая точность: {study.best_value:.4f}")

if __name__ == "__main__":
    os.makedirs(f"{PATH_TO_IMAGES}", exist_ok=True)

    main()