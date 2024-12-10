import pandas as pd
from sklearn.model_selection import train_test_split


def transform_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.set_index("Date")
    grouped = df.groupby('CodeProduit')
    # Dictionary to hold smaller dataframes for each product
    product_dataframes = {}
    # Loop through each group and create a smaller dataframe for each product
    for product, group in grouped:
        product_grouped = group.groupby(['Date', 'Pays'])['sales_number'].sum().reset_index()
        product_pivoted = product_grouped.pivot(index='Date', columns='Pays', values='sales_number')
        product_pivoted = product_pivoted.resample('D').sum().fillna(0)
        product_dataframes[product] = product_pivoted

    return product_dataframes


def preprocess_data(df, lags=3):
    df_lags = df.copy()
    for col in df.columns:
        for i in range(1, lags + 1):
            df_lags[f'{col}_lag_{i}'] = df_lags[col].shift(i)

    df_lags = df_lags.dropna()
    X_train, X_test, y_train, y_test = train_test_split(df_lags, df_lags, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test
