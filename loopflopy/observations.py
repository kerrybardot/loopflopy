import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import loopflopy.utils as utils

class Observations:
    def __init__(self, gdf, **kwargs):

        self.observations_label = "ObservationsBaseClass"
        self.gdf = gdf
        
        for key, value in kwargs.items():
            setattr(self, key, value)       
    
    def make_recarray(self):
       
        ### CREATE REC ARRAY FOR MODFLOW

        xobs = self.gdf.x.tolist()
        yobs = self.gdf.y.tolist()
        zobs = self.gdf.z.tolist()
        obslist = list(zip(xobs, yobs, zobs))

        # Cretae input arrays
        obscellid_list = self.gdf.id.tolist()
        obscell_list = self.gdf.cell_disu.tolist()

        obs_rec = []
        for i, cell_disu in enumerate(obscell_list):
            obs_rec.append([obscellid_list[i], 'head', (cell_disu+1)]) # 1 based for obs package

        self.obs_rec = obs_rec
        