"""
Module for training with hyper-paramater tuning and evaluating the 3 ML models
for the case of linear models, tree models, and boosting (ensemble) models.
"""

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

from src.model.ml.baseline import BaselineModel
from src.model.ml.evaluator import MLEvaluator
from src.preprocessing.dataset import DatasetBuilder


def run_ml_model(regressor, grid, model_name):
    # dataset
    dataset_builder = DatasetBuilder(n_in=3*24,
                                     granularity='1H',
                                     save_dataset=True,
                                     directory='../../data/datasets/hourly/df-3x24-just-steps.pkl',
                                     total_users=None)

    dataset = dataset_builder.create_dataset_all_features()

    x_train_val, _, y_train_val, _ = dataset_builder.get_train_test(dataset=dataset)
    x_train, x_val, x_test, y_train, y_val, y_test = dataset_builder.get_train_val_test(dataset=dataset)

    # Note: It is often beneficial to scale target variables too
    target_scaler = MinMaxScaler()

    # fit on train
    target_scaler.fit(y_train.values.reshape(-1, 1))

    # transform all targets
    y_train_val = target_scaler.transform(y_train_val.values.reshape(-1, 1))
    y_train = target_scaler.transform(y_train.values.reshape(-1, 1))
    y_val = target_scaler.transform(y_val.values.reshape(-1, 1))
    y_test = target_scaler.transform(y_test.values.reshape(-1, 1))

    # modeling
    baseline_ml = BaselineModel(regressor=regressor)

    # choose between simple train or cross-validate
    # model = baseline_ml.train_model(x=x_train, y=y_train)
    model = baseline_ml.tune_model(x=x_train_val, y=y_train_val.reshape(-1),
                                   grid_params=grid, path_to_save='../model/ml/models/{}.pkl'.format(model_name))

    evaluator = MLEvaluator(model)

    # inverse transformation for performance recording
    y_pred = evaluator.inference(x_test)
    y_pred_train = evaluator.inference(x_train)
    y_pred_val = evaluator.inference(x_val)

    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_pred_train = target_scaler.inverse_transform(y_pred_train.reshape(-1, 1))
    y_pred_val = target_scaler.inverse_transform(y_pred_val.reshape(-1, 1))

    y_test = target_scaler.inverse_transform(y_test)
    y_val = target_scaler.inverse_transform(y_val)
    y_train = target_scaler.inverse_transform(y_train)

    scores_test = evaluator.evaluate(y_test, y_pred)
    scores_train = evaluator.evaluate(y_train, y_pred_train)
    scores_val = evaluator.evaluate(y_val, y_pred_val)

    print("Train set scores:", scores_train)
    print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)

    evaluator.save_results(scores_train, '{}-train.csv'.format(model_name))
    evaluator.save_results(scores_val, '{}-val.csv'.format(model_name))
    evaluator.save_results(scores_test, '{}-test.csv'.format(model_name))


if __name__ == '__main__':
    MODELS = [
        Ridge(random_state=1),
        DecisionTreeRegressor(random_state=1),
        HistGradientBoostingRegressor(random_state=1)
    ]

    # Parameters for hyper-tuning
    ridge_grid = {
        "ridge__alpha": [0.5, 1.0, 2.0, 5.0],
        "ridge__tol": [1e-3, 1e-4, 1e-5],
        "ridge__solver": ["auto"]
    }

    trees_grid = {
        "decisiontreeregressor__criterion": ["mse", "friedman_mse"],
        "decisiontreeregressor__max_depth": [5, 10, 15, 20],
    }

    gb_grid = {
        "histgradientboostingregressor__loss": ['least_squares', 'poisson'],
        "histgradientboostingregressor__l2_regularization": [0, 1],
        "histgradientboostingregressor__max_iter": [100, 200, 300],
    }

    GRID_PARAMS = [
        ridge_grid,
        trees_grid,
        gb_grid
    ]

    MODEL_NAMES = [
        'ridge',
        'trees',
        'gb'
    ]

    for model, grid, model_name in zip(MODELS, GRID_PARAMS, MODEL_NAMES):
        run_ml_model(model, grid, model_name)
