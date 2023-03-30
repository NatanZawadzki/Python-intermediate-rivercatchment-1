import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas import sjoin

site_locations = pd.read_csv('data/LOCAR_Site_Information.csv',
                             skipfooter=11, engine='python',
                             usecols = ['Site Code', 'Longitude', 'Latitude'],
                             index_col='Site Code')

site_geometry = gpd.points_from_xy(site_locations['Longitude'],
                                   site_locations['Latitude'],
                                   crs='EPSG:4326')

site_gdf = gpd.GeoDataFrame(site_locations,geometry=site_geometry)
FP_catchment = gpd.GeoDataFrame.from_file('data/river_catchments/frome_piddle_catchment.shp')

FP_sites = sjoin(site_gdf,FP_catchment)


def is_site_within_catchment(site_dataframe, catchment_dataframe):
    answer_dataframe = sjoin(site_dataframe,catchment_dataframe)
    if answer_dataframe.size:
        return True
    else:
        return False

print(is_site_within_catchment(site_gdf.loc[['FP23']],FP_catchment))