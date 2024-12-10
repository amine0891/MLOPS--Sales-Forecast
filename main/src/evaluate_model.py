from sklearn.metrics import mean_squared_error


def evaluate_model(actual, pred):
    mse = mean_squared_error(actual, pred)
    return mse
