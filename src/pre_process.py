import pandas as pd
import geopandas as gpd


def filter_points_within_boundary(df, geojson_path):
    telangana_boundary = gpd.read_file(geojson_path)

    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")

    filtered_gdf = gdf_points[gdf_points.within(telangana_boundary.unary_union)]
    
    return filtered_gdf

def load_and_filter(geojson_file):
    df = pd.read_csv('clustering_data.csv', low_memory=False)

    telangana_data = df[df['StateName'] == 'TELANGANA']

    telangana_data.loc[:, 'Latitude'] = pd.to_numeric(telangana_data['Latitude'], errors='coerce')
    telangana_data.loc[:, 'Longitude'] = pd.to_numeric(telangana_data['Longitude'], errors='coerce')

    telangana_data = telangana_data.dropna(subset=['Latitude', 'Longitude'])

    telangana_data = filter_points_within_boundary(telangana_data, geojson_file)

    return telangana_data