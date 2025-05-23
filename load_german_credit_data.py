import pandas as pd

def load_german_credit_data_dataset():
    folder = "./german-credit-preprocessing"
    X_train_df = pd.read_csv(f'{folder}/german_X_train_scaled.csv')
    X_train = X_train_df.to_numpy().astype('float32')

    y_train_df = pd.read_csv(f'{folder}/german_y_train.csv')
    y_train = y_train_df['credit_rating'].to_numpy().astype('int32').reshape(-1)

    X_test_df = pd.read_csv(f'{folder}/german_X_test_scaled.csv')
    X_test = X_test_df.to_numpy().astype('float32')

    y_test_df = pd.read_csv(f'{folder}/german_y_test.csv')
    y_test = y_test_df['credit_rating'].to_numpy().astype('int32').reshape(-1)

    return X_train, y_train, X_test, y_test
