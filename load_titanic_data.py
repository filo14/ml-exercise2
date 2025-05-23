import pandas as pd

def load_titanic_dataset():
    folder = "./titanic-preprocessing"
    X_train_df = pd.read_csv(f'{folder}/titanic_X_train_scaled.csv')
    X_train = X_train_df.to_numpy().astype('float32')

    y_train_df = pd.read_csv(f'{folder}/titanic_y_train.csv')
    y_train = y_train_df['Survived'].to_numpy().astype('int32').reshape(-1)

    X_test_df = pd.read_csv(f'{folder}/titanic_X_test_scaled.csv')
    X_test = X_test_df.to_numpy().astype('float32')

    y_test_df = pd.read_csv(f'{folder}/titanic_y_test.csv')
    y_test = y_test_df['Survived'].to_numpy().astype('int32').reshape(-1)

    return X_train, y_train, X_test, y_test