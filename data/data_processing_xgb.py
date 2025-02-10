import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(time_lag=True, lag=3):
    """load data and create x_train, y_train, x_test, y_test for xgboost"""
    node_features = pd.read_csv('nodes_df.csv')
    ts = pd.read_csv('ts_dt_hourly.csv', )
    ts = ts.sort_values(by='datetime').reset_index(drop=True)

    #dropping the old index column
    ts.drop(columns=[ts.columns[0]], inplace=True)
    ref_temp_hourly = pd.read_csv('ref_temp_hourly.csv')
    ref_temp_hourly.sort_values(by='datetime', inplace=True)

    weather_data_hourly = pd.read_csv('weather_data.csv')
    weather_data_hourly.rename(columns={'time': 'datetime'}, inplace=True)
    #fill missing values with the median of the row
    #ts.fillna(ref_temp_hourly.drop(columns=['datetime']).median(), inplace=True)

    #drop missing temp values
    #ts.dropna(inplace=True)


    df_timeseries_melted = pd.melt(ts, id_vars=['datetime'], 
                               var_name='Log', value_name='temperature_delta')
    df_timeseries_melted['Log_Nr'] = df_timeseries_melted['Log'].str.replace('Log_', '').astype(int)

    df_combined = pd.merge(df_timeseries_melted, node_features, on='Log_Nr', how='left')
    df_combined = pd.merge(df_combined, ref_temp_hourly, on='datetime', how='left')
    df_combined = pd.merge(df_combined, weather_data_hourly, on='datetime', how='left')
    #df_combined.dropna(subset=['temperature_delta'], inplace=True)

    #convert the datetime column to datetime object
    df_combined['datetime'] = pd.to_datetime(df_combined['datetime'])

    train_df = df_combined[df_combined['datetime'].dt.year < 2022]
    test_df = df_combined[df_combined['datetime'].dt.year >= 2022]

    #add some features
    train_df['hour'] = train_df['datetime'].dt.hour
    train_df['month'] = train_df['datetime'].dt.month

    test_df['hour'] = test_df['datetime'].dt.hour
    test_df['month'] = test_df['datetime'].dt.month

    #change hour into cyclical
    train_df['hour_sin'] = np.sin(train_df['hour']*(2.*np.pi/24))
    train_df['hour_cos'] = np.cos(train_df['hour']*(2.*np.pi/24))

    test_df['hour_sin'] = np.sin(test_df['hour']*(2.*np.pi/24))
    test_df['hour_cos'] = np.cos(test_df['hour']*(2.*np.pi/24))

    test_df.drop(columns=['hour'], inplace=True)
    train_df.drop(columns=['hour'], inplace=True)

    #train_df['is_daytime'] = train_df['datetime'].apply(is_daytime)
    #test_df['is_daytime'] = test_df['datetime'].apply(is_daytime)

    #add temporal lag for temperature delta
    if time_lag:
        train_df =train_df.groupby('Log_Nr', group_keys=False).apply(lambda group: add_temporal_lag(group, lag=lag))
        test_df = test_df.groupby('Log_Nr', group_keys=False).apply(lambda group: add_temporal_lag(group, lag=lag))

    train_df.drop(columns=['Unnamed: 0', 'Log'], inplace=True)
    test_df.drop(columns=['Unnamed: 0', 'Log'], inplace=True)

    X_train = train_df.drop(columns=['temperature_delta', 'datetime', 'Log_Nr'])
    X_test = test_df.drop(columns=['temperature_delta', 'datetime', 'Log_Nr'])


    Y_train = train_df['temperature_delta']
    Y_test = test_df['temperature_delta']

    nodes_test_index = test_df['Log_Nr'].values
    nodes_train_index = train_df['Log_Nr'].values

    print(len(Y_train), len(nodes_train_index))



    return X_train, Y_train, X_test, Y_test, nodes_test_index, nodes_train_index

def is_daytime(datetime):
    """check if it is daytime"""
    if datetime.hour >= 6 and datetime.hour <= 20:
        return 1
    else:
        return 0


def add_temporal_lag(df, lag=3):
    """add temporal lag to temperature delta"""
    if type(lag) == int:
        for i in range(1, lag+1):
            df[f'temperature_delta_lag_{i}'] = df['temperature_delta'].shift(i).fillna(method='bfill')
        return df
    elif type(lag) == list:
        for i in lag:
            df[f'temperature_delta_lag_{i}'] = df['temperature_delta'].shift(i).fillna(method='bfill')
        return df