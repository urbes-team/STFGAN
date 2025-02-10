import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_cluster import knn_graph
import geopandas as gpd
import torch
from tqdm import tqdm
def load_data(edges_type='knn', n_neighbors=10):
    """load data and create x_train, y_train, x_test, y_test for xgboost"""
    node_features = pd.read_csv('nodes_df.csv') #shape : (n_nodes, n_features)
    node_geom = gpd.read_file('buffered_stations.shp')
    ts = pd.read_csv('ts_dt_hourly.csv')
    ts = ts.sort_values(by='datetime').reset_index(drop=True)
    node_features = node_features[node_features['Log_Nr'] != 98]

    node_geom = node_geom[node_geom['Log_Nr'].isin(node_features['Log_Nr'])]
    node_geom = node_geom.set_crs('epsg:2056')

    node_geom['centroid'] = node_geom.centroid
    node_geom['y'] = node_geom['centroid'].y
    node_geom['x'] = node_geom['centroid'].x

    columns_to_keep = ['Log_'+ str(i) for i in node_features['Log_Nr'].unique()]
    #dropping the old index column
    ts.drop(columns=[ts.columns[0]], inplace=True)
    ref_temp_hourly = pd.read_csv('ref_temp_hourly.csv')
    ref_temp_hourly.sort_values(by='datetime', inplace=True)
    weather_data_hourly = pd.read_csv('weather_data.csv')
    weather_data_hourly.rename(columns={'time':'datetime'}, inplace=True)
    #fill missing values with the median of the row
    train_mask = ~ts[columns_to_keep].isna().to_numpy()
    # Calculate row-wise medians, excluding 'datetime'
    row_medians = ts.drop(columns=['datetime']).median(axis=1)

    # Fill NaN values row-wise
    ts.iloc[:, 1:] = ts.iloc[:, 1:].apply(
        lambda row: row.fillna(row_medians[row.name]), axis=1
    )
    ts['datetime'] = pd.to_datetime(ts['datetime'])
    ts['hour'] = ts['datetime'].dt.hour
    ts['month'] = ts['datetime'].dt.month

    #drop rows with at least one na
    print(ts.isna().sum().sum())


    #change hour into cyclical values
    ts['hour_sin'] = np.sin(2 * np.pi * ts['hour']/24.0)
    ts['hour_cos'] = np.cos(2 * np.pi * ts['hour']/24.0)

    ts.drop(columns=['hour'], inplace=True)

    mask = ts['datetime'].dt.year < 2022


    test_mask = train_mask[~mask]
    train_mask = train_mask[mask]

    #build edges
    if edges_type == 'knn':
        edges = knn_graph(torch.Tensor(node_features[['population', 'avg_height', 'elevation', 'dist_to_center', 'dist_to_forest']].values), k=n_neighbors, batch=None, loop=True)
    elif edges_type == 'distance':
        # Get k nearest neighbors based on geometric coordinates
        coords = torch.Tensor(node_geom[['x', 'y']].values)
        edges = knn_graph(coords, k=n_neighbors, batch=None, loop=True)

    elif edges_type == 'all':
        # Connect all nodes to all other nodes
        n_nodes = len(node_features)
        rows, cols = torch.meshgrid(node_features.index, node_features.index)
        edges = torch.stack([rows.flatten(), cols.flatten()], dim=0)

    elif edges_type == 'mix':
        # Get k nearest neighbors based on geometric distance
        coords = torch.Tensor(node_geom[['x', 'y']].values)
        dist_edges = knn_graph(coords, k=10, batch=None, loop=True)

        # Get k nearest neighbors based on features
        feature_edges = knn_graph(torch.Tensor(node_features[['population', 'avg_height', 'elevation', 'dist_to_center', 'dist_to_forest']].values), k=10, batch=None, loop=True)

        # Combine both edge sets
        edges = torch.cat([dist_edges, feature_edges], dim=1)
        # Remove duplicates
        edges = torch.unique(edges, dim=1)
    else:
        raise ValueError('edges_type must be knn')

    temporal_features_train = torch.Tensor(ts.loc[mask][['hour_sin','hour_cos', 'month']].values) #shape : (n_samples, n_features)
    temporal_features_train = torch.cat([temporal_features_train, torch.Tensor(ref_temp_hourly.loc[mask]['Log_98'].values.reshape(-1, 1))], dim=1)
    temporal_features_train = torch.cat([temporal_features_train, torch.Tensor(weather_data_hourly.drop(columns=['datetime']).loc[mask].values)], dim=1)

    temporal_features_test = torch.Tensor(ts.loc[~mask][['hour_sin','hour_cos','month']].values) #shape : (n_samples, n_features)
    temporal_features_test = torch.cat([temporal_features_test, torch.Tensor(ref_temp_hourly.loc[~mask]['Log_98'].values.reshape(-1, 1))], dim=1)
    temporal_features_test = torch.cat([temporal_features_test, torch.Tensor(weather_data_hourly.drop(columns=['datetime']).loc[~mask].values)], dim=1)

    #scale continous node features
    scaler = StandardScaler()
    node_features[['population','avg_height', 'elevation', 'd10_avg',
       'd_25_avg', 'd_50_avg', 'Albedo', 'NDVI', 'build_coun', 'dist_to_center']] = scaler.fit_transform(node_features[['population','avg_height', 'elevation', 'd10_avg',
                                                                                                                        'd_25_avg', 'd_50_avg', 'Albedo', 'NDVI', 'build_coun', 'dist_to_center']])


    #encode categorical features
    node_features = pd.get_dummies(node_features, columns=['LC', 'LCZ'], )
    node_features.drop(columns=['Log_Nr'])
    nodes = torch.Tensor(node_features.to_numpy(dtype=float)) #shapes : (n_nodes, n_features)

    target_train = torch.Tensor(ts.loc[mask][columns_to_keep].values) #shape : (n_samples, n_nodes)

    target_test = torch.Tensor(ts.loc[~mask][columns_to_keep].values) #shape : (n_samples, n_nodes)
    return nodes, edges, temporal_features_train, temporal_features_test, target_train, target_test, train_mask, test_mask