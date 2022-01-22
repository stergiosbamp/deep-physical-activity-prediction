from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

from src.model.ml.baseline import BaselineModel
from src.model.ml.evaluator import MLEvaluator
from src.preprocessing.dataset import DatasetBuilder


def trees():
    trees_grid = {
        "decisiontreeregressor__criterion": ["mse", "friedman_mse"],
        "decisiontreeregressor__max_depth": [5, 10, 15, 20],
    }

    trees_regressor = DecisionTreeRegressor(random_state=1)

    baseline_ml = BaselineModel(x_train=x_train, x_val=x_val, x_test=x_test,
                                y_train=y_train, y_val=y_val, y_test=y_test,
                                regressor=trees_regressor)

    baseline_ml.tune_model(X=x_train_val, y=y_train_val, grid_params=trees_grid,
                           path_to_save="../model/ml/models/tree.pkl")

    scores_train = baseline_ml.evaluator.evaluate_train()
    scores_val = baseline_ml.evaluator.evaluate_val()
    scores_test = baseline_ml.evaluator.evaluate_test()

    print("Train set scores:", scores_train)
    print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)

    baseline_ml.evaluator.save_results(scores_train, "trees-train.csv")
    baseline_ml.evaluator.save_results(scores_val, "trees-val.csv")
    baseline_ml.evaluator.save_results(scores_test, "trees-test.csv")

    baseline_ml.evaluator.plot_predictions_train(smooth=True)
    baseline_ml.evaluator.plot_predictions(smooth=True)


def boosting():
    gb_grid = {
        "histgradientboostingregressor__loss": ['least_squares', 'poisson'],
        "histgradientboostingregressor__l2_regularization": [0, 1],
        "histgradientboostingregressor__max_iter": [100, 200, 300],
    }

    gb_regressor = HistGradientBoostingRegressor(random_state=1)

    baseline_ml = BaselineModel(x_train=x_train, x_val=x_val, x_test=x_test,
                                y_train=y_train, y_val=y_val, y_test=y_test,
                                regressor=gb_regressor)

    baseline_ml.tune_model(X=x_train_val, y=y_train_val, grid_params=gb_grid,
                           path_to_save="../model/ml/models/gb.pkl")

    scores_train = baseline_ml.evaluator.evaluate_train()
    scores_val = baseline_ml.evaluator.evaluate_val()
    scores_test = baseline_ml.evaluator.evaluate_test()

    print("Train set scores:", scores_train)
    print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)

    baseline_ml.evaluator.save_results(scores_train, "gb-train.csv")
    baseline_ml.evaluator.save_results(scores_val, "gb-val.csv")
    baseline_ml.evaluator.save_results(scores_test, "gb-test.csv")

    baseline_ml.evaluator.plot_predictions_train(smooth=True)
    baseline_ml.evaluator.plot_predictions(smooth=True)


if __name__ == '__main__':
    # dataset
    dataset_builder = DatasetBuilder(n_in=3*24,
                                     granularity='whatever',
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
    ridge_grid = {
        "ridge__alpha": [0.5, 1.0, 2.0, 5.0],
        "ridge__tol": [1e-3, 1e-4, 1e-5],
        "ridge__solver": ["auto"]
    }
    ridge_regressor = Ridge(random_state=1)

    baseline_ml = BaselineModel(regressor=ridge_regressor)

    # choose between simple train or cross-validate
    # model = baseline_ml.train_model(x=x_train, y=y_train)
    model = baseline_ml.tune_model(x=x_train_val, y=y_train_val, grid_params=ridge_grid)

    evaluator = MLEvaluator(model)

    y_pred = evaluator.inference(x_test)

    # inverse transform for performance recording
    y_test = target_scaler.inverse_transform(y_test)
    y_pred = target_scaler.inverse_transform(y_pred)

    # scores_train = baseline_ml.evaluator.evaluate(x_train, y_train)
    # scores_val = baseline_ml.evaluator.evaluate(x_val, y_val)
    scores_test = evaluator.evaluate(y_test, y_pred)

    # print("Train set scores:", scores_train)
    # print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)
