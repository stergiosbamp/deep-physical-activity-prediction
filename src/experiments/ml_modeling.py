from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from src.model.ml.baseline import BaselineModel
from src.preprocessing.dataset import DatasetBuilder


def ridge():
    ridge_grid = {
        "ridge__alpha": [0.5, 1.0, 2.0, 5.0],
        "ridge__tol": [1e-3, 1e-4, 1e-5],
        "ridge__solver": ["auto"]
    }
    ridge_regressor = Ridge(random_state=1)

    baseline_ml = BaselineModel(x_train=x_train, x_val=x_val, x_test=x_test,
                                y_train=y_train, y_val=y_val, y_test=y_test,
                                regressor=ridge_regressor)

    baseline_ml.tune_model(X=x_train_val, y=y_train_val, grid_params=ridge_grid)

    scores_train = baseline_ml.evaluator.evaluate_train()
    scores_val = baseline_ml.evaluator.evaluate_val()
    scores_test = baseline_ml.evaluator.evaluate_test()

    print("Train set scores:", scores_train)
    print("Val set scores:", scores_val)
    print("Test set scores:", scores_test)

    baseline_ml.evaluator.save_results(scores_train, "ridge-train.csv")
    baseline_ml.evaluator.save_results(scores_val, "ridge-val.csv")
    baseline_ml.evaluator.save_results(scores_test, "ridge-test.csv")

    baseline_ml.evaluator.plot_predictions_train(smooth=True)
    baseline_ml.evaluator.plot_predictions(smooth=True)


def trees():
    trees_grid = {
        "decisiontreeregressor__criterion": ["mse", "friedman_mse"],
        "decisiontreeregressor__max_depth": [5, 10, 15, 20],
    }

    trees_regressor = DecisionTreeRegressor(random_state=1)

    baseline_ml = BaselineModel(x_train=x_train, x_val=x_val, x_test=x_test,
                                y_train=y_train, y_val=y_val, y_test=y_test,
                                regressor=trees_regressor)

    baseline_ml.tune_model(X=x_train_val, y=y_train_val, grid_params=trees_grid)

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

    baseline_ml.tune_model(X=x_train_val, y=y_train_val, grid_params=gb_grid)

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
    dataset_builder = DatasetBuilder(n_in=3*24,
                                     granularity='whatever',
                                     save_dataset=True,
                                     directory='../../data/datasets/hourly/df-3x24-just-steps.pkl',
                                     total_users=None)

    dataset = dataset_builder.create_dataset_all_features()
    x_train_val, _, y_train_val, _ = dataset_builder.get_train_test(dataset=dataset)
    x_train, x_val, x_test, y_train, y_val, y_test = dataset_builder.get_train_val_test(dataset=dataset)

    # ridge()
    trees()
    # boosting()
