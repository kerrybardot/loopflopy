import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

class Observations:
    def __init__(self, df_boredetails, **kwargs):

        self.observations_label = "ObservationsBaseClass"
        self.df_boredetails = df_boredetails

        for key, value in kwargs.items():
            setattr(self, key, value)       
    
    def process_obs(self, spatial, geomodel, mesh):

        # Get observation elevation (z) from dataframe
        '''depth = spatial.obsbore_gdf.zobs_mbgl.tolist()
        zobs = []
        for n in range(spatial.nobs):
            icpl = mesh.obs_cells[n]
            zobs.append(geomodel.top_geo[icpl] - depth[n])'''
        
        zobs = spatial.obsbore_gdf.zobs.tolist()
        xobs, yobs = spatial.obsbore_gdf.Easting.tolist(), spatial.obsbore_gdf.Northing.tolist(), 
        obslist = list(zip(xobs, yobs, zobs))
    
        # Cretae input arrays
        obs_rec = []
        for i, cell in enumerate(mesh.obs_cells):
            x,y,z = obslist[i][0], obslist[i][1], obslist[i][2]
            point = Point(x,y,z)
            lay, icpl = geomodel.vgrid.intersect(x,y,z)
            cell_disv = icpl + lay*mesh.ncpl
            cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
            obs_rec.append([spatial.idobsbores[i], 'head', (cell_disu+1)])   
    
        self.obs_rec = obs_rec
        