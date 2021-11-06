import os
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.preprocessing import MinMaxScaler

from src.model.dl.lstm import LSTMRegressor
from src.config.directory import BASE_PATH_HOURLY_DATASETS
from src.preprocessing.dataset import DatasetBuilder


if __name__ == '__main__':
    p = dict(
        batch_size=128,
        criterion=nn.L1Loss(),
        max_epochs=30,
        n_features=72,
        hidden_size=100,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
        num_workers=4
    )

    # Load PyTorch model
    loaded = LSTMRegressor.load_from_checkpoint(
        '/home/stergios/Development/MSc/Thesis/deep-physical-activity-prediction/src'
        '/model/tb_logs/my_model/version_24/checkpoints/epoch=29-step=7559.ckpt',
        n_features=p['n_features'],
        hidden_size=p['hidden_size'],
        batch_size=p['batch_size'],
        criterion=p['criterion'],
        num_layers=p['num_layers'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate'])

    # Get the dataset to get the same train/test splits
    ds_builder = DatasetBuilder(n_in=3 * 24,
                                granularity='whatever',
                                save_dataset=True,
                                directory=os.path.join('/home/stergios/Development/MSc/Thesis/deep-physical-activity'
                                                       '-prediction/data/datasets/hourly/df-3*24-imputed-no-outliers-all-features-all-users-with-subject-injected.pkl'))

    dataset = ds_builder.create_dataset_steps_features()
    x_train, x_test, y_train, y_test = ds_builder.get_train_test(dataset=dataset)

    # Scale the test set
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Inference from PyTorch model and get predictions as numpy array
    y_pred = loaded(torch.from_numpy(x_test).float())

    y_pred = y_pred.detach().numpy()

    # Now use predictions with any metric from sklearn
    print("MAE", mean_absolute_error(y_test, y_pred))
    print("R2", r2_score(y_test, y_pred))
    print("MdAE", median_absolute_error(y_test, y_pred))
