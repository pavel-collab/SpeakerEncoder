# Run the optuna test

To suggest the most optimal hyperparameters run the optuna test:
```
python3 ./tests/optuna_test.py optuna.n_trials=50 train.n_epochs=50
```
Check out the parameters n_trials and n_epochs according to your computer and time resources.

# Run the model training

When you suggest your model hyperparameters, you can run the training of final model
```
python3 train.py
```

Set the nesessary hyperparameters via command line or hydra config file.