import pandas as pd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import matplotlib.pyplot as plt
import geopandas as gpd
import loopflopy.utils as utils
import math
import numpy as np

def make_obs_df(mesh, geomodel, nobs_x, nobs_z):
    """
    Note: These are "fake" obs bores. Code needs to be upgrades to project real obs bore 
    coords onto transect.
    Create a DataFrame of observation points based on the mesh and geomodel.
    Observation points are sampled from the cell centres in the x-y plane and
    vertically distributed between z0 and z1 of the geomodel.
    """

    # Make fake observation points by sampling cell centres (saves interpolating later!)
    a = np.linspace(0, len(mesh.xcyc)-1, nobs_x+2, dtype=int)[1:-1]
    x = [mesh.xcyc[i][0] for i in a]  # Extract x coordinates directly  
    y = [mesh.xcyc[i][1] for i in a]  # Extract y coordinates directly

    # Making observation points in vertical direction
    z = np.linspace(geomodel.z1, geomodel.z0, nobs_z+2)[1:-1] # Don't include edges! 

    obs_points = []
    for i in range(len(x)):
        for j in range(len(z)):
            obs_points.append([x[i], y[i], z[j]]) 
    print(len(obs_points)) # Should be nobs_x * nobs_z
    print(obs_points)

    X,Y,Z = list(zip(*obs_points))

    # Create DataFrame
    df = pd.DataFrame({
        'id': [str('obs_%i' %i) for i in np.arange(len(X))],
        'x': X,
        'y': Y,
        'z': Z,
    })
    return df

def make_obs_gdf(df, geomodel, mesh, spatial):

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=spatial.epsg)

    gdf['icpl'] = gdf.apply(lambda row: mesh.vgrid.intersect(row.x,row.y), axis=1)
    gdf['ground'] = gdf.apply(lambda row: geomodel.top[row.icpl], axis=1)
    gdf['model_bottom'] = gdf.apply(lambda row: geomodel.botm[-1, row.icpl], axis=1)
    gdf['z-bot'] = gdf.apply(lambda row: row['z'] - row['model_bottom'], axis=1)

    for idx, row in gdf.iterrows():
        result = row['z'] - row['model_bottom']
        if result < 0:
            print(f"Bore {row['id']} has an elevation below model bottom by: {result} m, removing from obs list")

    gdf = gdf[gdf['z-bot'] > 0] # filters out observations that are below the model bottom
    gdf['cell_disv'] = gdf.apply(lambda row: utils.xyz_to_disvcell(geomodel, row.x, row.y, row.z), axis=1)
    gdf['cell_disu'] = gdf.apply(lambda row: utils.disvcell_to_disucell(geomodel, row['cell_disv']), axis=1)  
    gdf['(lay,icpl)'] = gdf.apply(lambda row: utils.disvcell_to_layicpl(geomodel, row['cell_disv']), axis = 1)
    gdf['lay']        = gdf.apply(lambda row: row['(lay,icpl)'][0], axis = 1)
    gdf['icpl']       = gdf.apply(lambda row: row['(lay,icpl)'][1], axis = 1)
    gdf['obscell_xy'] = gdf['icpl'].apply(lambda icpl: (mesh.xcyc[icpl][0], mesh.xcyc[icpl][1]))
    gdf['obscell_z']  = gdf.apply(lambda row: geomodel.zc[row['lay'], row['icpl']], axis=1)
    gdf['obs_zpillar']  = gdf.apply(lambda row: geomodel.zc[:, row['icpl']], axis=1)
    if 'nls' in geomodel.__dict__:
        gdf['geolay']       = gdf.apply(lambda row: math.floor(row['lay']/geomodel.nls), axis = 1) # model layer to geolayer

    # Make sure no pinched out observations
    if -1 in gdf['cell_disu'].values:
        print('Warning: some observations are pinched out. Check the model and data.')
        print('Number of pinched out observations removed: ', len(gdf[gdf['cell_disu'] == -1]))
        gdf = gdf[gdf['cell_disu'] != -1] # delete pilot points where layer is pinched out

    return gdf
