import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import loopflopy.utils as utils
import math

# Now we have got extra bore details from WIR, we can groundlevel and well screens to our dataframe
def make_df():
    # Add GL to main df
    df = pd.read_excel('../data/example_data.xlsx', sheet_name='obs_bores')
    df = df.drop(columns=['Depth From/To (mbGL)'])
    df = df.rename(columns={'Site Short Name': 'ID'})
    df_boredetails = df
    return df_boredetails 


def plot_hydrographs(df_obs):
    # Plot water levels - Yarragadee Aquifer
    yarr_df = df_obs[df_obs['Aquifer Name'] == 'Perth-Yarragadee North']
    yarr_bores = yarr_df['Site Ref'].unique()

    for bore in yarr_bores:
        df = yarr_df[yarr_df['Site Ref'] == bore]
        plt.plot(df['Collect Date'], df['Reading Value'], label = df['ID'].iloc[0])
    plt.legend(loc = 'upper left',fontsize = 'small', markerscale=0.5)

def make_obs_gdf(df, geomodel, mesh, spatial):
    
    df = df.rename(columns={'Easting': 'x', 'Northing': 'y'})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=spatial.epsg)

    # Check if points are within the model boundary polygon
    mask = gdf.geometry.within(spatial.model_boundary_poly)
    if not mask.all():
        print("The following geometries are NOT within the polygon:")
        print(gdf[~mask])
    else:
        print("All geometries are within the polygon.")
    gdf = gdf[gdf.geometry.within(spatial.model_boundary_poly)] # Filter points outside model

    gdf = gdf[~gdf['z'].isna()] # Don't include obs with no zobs
    gdf = gdf[gdf['z'] > geomodel.z0] # Don't include obs deeper than flow model bottom
    gdf = gdf.reset_index(drop=True)

    # Perform the intersection
    gdf['icpl'] = gdf.apply(lambda row: mesh.vgrid.intersect(row.x,row.y), axis=1)
    gdf['ground'] = gdf.apply(lambda row: geomodel.top_geo[row.icpl], axis=1)
    gdf['model_bottom'] = gdf.apply(lambda row: geomodel.botm[-1, row.icpl], axis=1)
    gdf['z-bot'] = gdf.apply(lambda row: row['z'] - row['model_bottom'], axis=1)

    for idx, row in gdf.iterrows():
        result = row['z'] - row['model_bottom']
        if result < 0:
            print(f"Bore {row['id']} has a z elevation below model bottom by: {result} m, removing from obs list")

    gdf = gdf[gdf['z-bot'] > 0] # filters out observations that are below the model bottom

    gdf['cell_disv'] = gdf.apply(lambda row: utils.xyz_to_disvcell(geomodel, row.x, row.y, row.z), axis=1)
    gdf['cell_disu'] = gdf.apply(lambda row: utils.disvcell_to_disucell(geomodel, row['cell_disv']), axis=1)  

    gdf['(lay,icpl)'] = gdf.apply(lambda row: utils.disvcell_to_layicpl(geomodel, row['cell_disv']), axis = 1)
    gdf['lay']        = gdf.apply(lambda row: row['(lay,icpl)'][0], axis = 1)
    gdf['icpl']       = gdf.apply(lambda row: row['(lay,icpl)'][1], axis = 1)
    gdf['obscell_xy'] = gdf['icpl'].apply(lambda icpl: (mesh.xcyc[icpl][0], mesh.xcyc[icpl][1]))
    gdf['obscell_z']  = gdf.apply(lambda row: geomodel.zc[row['lay'], row['icpl']], axis=1)
    gdf['obs_zpillar']  = gdf.apply(lambda row: geomodel.zc[:, row['icpl']], axis=1)
    gdf['geolay']       = gdf.apply(lambda row: math.floor(row['lay']/geomodel.nls), axis = 1) # model layer to geolayer

    # Make sure no pinched out observations
    if -1 in gdf['cell_disu'].values:
        print('Warning: some observations are pinched out. Check the model and data.')
        print('Number of pinched out observations: ', len(gdf[gdf['cell_disu'] == -1]))
        gdf = gdf[gdf['cell_disu'] != -1] # delete pilot points where layer is pinched out

    return gdf