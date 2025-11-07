from models import ModelParams, ModuleParams, SpeakerDataset, Conformer, AngularMarginSoftmax, evaluate
from logger import local_logger, LOG_PATH

from dataclasses import replace

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import random
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

N_EPOCH = 50

def fix_torch_seed(seed: int = 42) -> None:
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def train(conf: ModelParams) -> float:
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
    
    #! разные варианты шедулеров для экспериментов
    #TODO: возможно стоит вынести это в параметры обучения и передавать в конфиге
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optim,
    #     max_lr=conf.learning_rate,
    #     epochs=conf.n_epochs,
    #     steps_per_epoch=len(train_dataloader),
    #     pct_start=0.1
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, 
        T_max=100,  # Количество итераций до минимального LR
        eta_min=conf.learning_rate  # Минимальный learning rate
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optim,
    #     T_0=50,        # Количество итераций до первого рестарта
    #     T_mult=2,      # Множитель для увеличения T_0 после каждого рестарта
    #     eta_min=conf.learning_rate
    # )

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
            scheduler.step()

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
        local_logger.info(f"Epoch {epoch + 1}: Avg Train Loss = {avg_epoch_loss:.4f}, Train Accuracy = {epoch_accuracy:.4f}")

        if val_dataloader and (epoch + 1) % conf.validation_frequency == 0:
            local_logger.info(f"\nRunning validation evaluation at epoch {epoch + 1}...")
            eer = evaluate(model, val_dataloader, conf.device)
            writer.add_scalar("Validation/EER", eer, epoch)
            local_logger.info(f"Validation EER = {eer:.4f}")

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
                local_logger.info(f"Saved best model with EER: {best_eer:.4f}")

    writer.close()
    return best_eer

def objective(trial):
    time_log_marker = datetime.today().strftime("%d_%m_%Y_%H_%M")

    # Определяем пространство поиска гиперпараметров

    # Параметры для обучения с AngularMarginSoftmax
    # Для AngularMarginSoftmax часто требуются специфические значения margin и scale.
    # Типичные значения для ArcFace: m=0.5, s=64.
    #TODO: по хорошему, стоит разделить параметры модели и параметры эксперимента
    params_am = ModuleParams(
        dataset_dir=Path("./data/train"),
        use_cache=False, # Можно установить в True, если датасет полностью помещается в RAM.
        checkpoints_dir=Path("./checkpoints"),
        model_params=ModelParams(
            emb_size=trial.suggest_categorical("emb_size", [16, 32, 64, 128, 256, 512])
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        # num_workers=4 if torch.cuda.is_available() else 1,
        num_workers=4,
        n_epochs=N_EPOCH, # Увеличиваем количество эпох, т.к. AM-Softmax сходится медленнее
        log_dir=Path(f"{LOG_PATH}/angular_margin_{time_log_marker}"),
        loss_function="angular_margin",
        angular_margin=trial.suggest_float("angular_margin", 0.1, 0.9, log=True),
        angular_scale=trial.suggest_float("angular_scale", 16.0, 64.0),
        validation_dir=Path("./data/dev"),
        validation_frequency=1,
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-3),
        batch_size=4
    )

    local_logger.info(f"Starting training: tensorboard time log marker: {time_log_marker}")
    local_logger.info(f"Current model params: {params_am}")
    best_err = train(params_am)
    local_logger.info("Training completed")
    
    return best_err

def fix_seed(func):
    def wrapper(*args, **kwargs):
        fix_torch_seed(seed=43)
        return func(*args, **kwargs)
    return wrapper