import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import loopflopy.utils as utils

class Observations:
    def __init__(self, df_boredetails, **kwargs):

        self.observations_label = "ObservationsBaseClass"
        self.df_boredetails = df_boredetails

        for key, value in kwargs.items():
            setattr(self, key, value)       
    
    def process_obs(self, spatial, geomodel, mesh):
       
        zobs = spatial.obsbore_gdf.zobs.tolist()
        xobs, yobs = spatial.obsbore_gdf.Easting.tolist(), spatial.obsbore_gdf.Northing.tolist(), 
        obslist = list(zip(xobs, yobs, zobs))
    
        # Cretae input arrays
        obs_rec = []
        for i, cell in enumerate(mesh.obs_cells):
            x,y,z = obslist[i][0], obslist[i][1], obslist[i][2]
            try:
                cell_disu = utils.xyz_to_disucell(geomodel, x,y,z)
                if cell_disu != -1:
                    obs_rec.append([spatial.idobsbores[i], 'head', (cell_disu+1)]) 
                else:
                    print('pinched out!') # that shoudl not happen!  
            except:
                print('No obs at (%f, %f, %f)' %(x,y,z))

        self.obs_rec = obs_rec
        