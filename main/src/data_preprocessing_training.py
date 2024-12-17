import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
import math

def fill_missing_dates(df, start_date=None, end_date=None):


    # Determine the date range
    if not start_date:
        start_date = df['Date'].min()
    if not end_date:
        end_date = df['Date'].max()

    # Generate a complete date range
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a cartesian product of all dates with product_code and country
    product_country = df[['CodeProduit', 'Pays']].drop_duplicates()
    complete_index = pd.DataFrame(
        [(product, country, date) for date in all_dates
                                   for product, country in product_country.values],
        columns=['CodeProduit', 'Pays', 'Date']
    )

    # Merge the original data with the complete index (based on product_code, country, and date)
    df_complete = pd.merge(complete_index, df, on=['CodeProduit', 'Pays', 'Date'], how='left')

    # Fill missing sales_number with 0
    df_complete['sales_number'] = df_complete['sales_number'].fillna(0)

    # Set the 'date' column as the index again
    df_complete.set_index('Date', inplace=True)

    return df_complete

def encode_data(df, encoder_exists=False):
    encoder = None
    if encoder_exists:
        with open('resources/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
    else:
        # Initialize and fit the encoder
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
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

def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = fill_missing_dates(df)
    data_lags = preprocess_data(df)
    encoded_data = encode_data(data_lags)
    return encoded_data



def preprocess_data(df, lags=15):
    df_lags = df.copy()
    for i in range(1, lags+1):
        df_lags[f'sales_number_lag_{i}'] = df_lags.groupby(['CodeProduit', 'Pays'])['sales_number'].shift(i)
    df_lags = df_lags.fillna(0)
    return df_lags


def process_and_split_data(df):
    preprocessed_data = prepare_data(df)
    X = preprocessed_data.drop(columns=["sales_number"])
    y = preprocessed_data["sales_number"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    return X_train, X_test, y_train, y_test

def predict_for_future_date(df, product_code, country, target_date, model):

    product_country_data = df[(df['CodeProduit'] == product_code) & (df['Pays'] == country)]
    product_country_data.index = pd.to_datetime(product_country_data.index)
    product_country_data = preprocess_data(product_country_data)
    latest_data = product_country_data[product_country_data.index <= target_date].tail(15)
    current_date = latest_data.index.values[-1]

    forecast_results = []
    while current_date < target_date:
        # Get the current sample for prediction (including lag features)
        current_sample = latest_data.tail(1)
        current_sample = encode_data(current_sample, encoder_exists=True)
        current_sample = current_sample.drop("sales_number", axis=1)
        predicted_sales = model.predict(current_sample)
        forecast_results.append({
            'Date': current_date,
            'predicted_sales': round(predicted_sales[0])
        })

        # Step 6: Update the dataset for the next day's forecast
        new_date = current_date + pd.Timedelta(days=1)

        # Create a new row and set the index as the new date
        new_row = pd.DataFrame({
            'CodeProduit': [product_code],
            'Pays': [country],
            'sales_number': [predicted_sales[0]],
            **{
                f'sales_number_lag_{i}': [latest_data['sales_number'].values[-i]]
                for i in range(1, 16)
            }
        }, index=[new_date])

        # Add the new row to the data
        latest_data = pd.concat([latest_data, new_row])

        # Move to the next day
        latest_data = latest_data.drop(latest_data.index[0])
        current_date = latest_data.index.values[-1]
        
    return forecast_results
