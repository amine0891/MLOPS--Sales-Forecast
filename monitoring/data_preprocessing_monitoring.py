import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

def transform_data(df, encoder_exists=False):
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.set_index("Date")
    encoder = None
    if encoder_exists:
        with open('resources/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
    else:
        # Initialize and fit the encoder
        encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
        encoder.fit(df[['Pays', 'CodeProduit']])
        with open('resources/encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)

    # Transform data using the same feature set
    encoded_data = encoder.transform(df[['Pays', 'CodeProduit']])

    # Create encoded DataFrame with feature names
    encoded_columns = encoder.get_feature_names_out(['Pays', 'CodeProduit'])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns,
        index=df.index
    )

    # Combine encoded data with the original DataFrame
    data = pd.concat([df, encoded_df], axis=1)

    # Drop original columns
    data.drop(columns=['Pays', 'CodeProduit'], inplace=True)
    return data


def preprocess_data(df, lags=5):
    df_lags = df.copy()
    for i in range(1, lags + 1):
        df_lags[f'sales_number_lag_{i}'] = df_lags['sales_number'].shift(i)

    df_lags = df_lags.fillna(0)
    return df_lags

