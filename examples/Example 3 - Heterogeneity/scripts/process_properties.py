import pandas as pd
import numpy as np
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import loopflopy.utils as utils
import pickle
from scipy.interpolate import griddata

class Properties:
    def __init__(self):

            self.data_label = "DataBaseClass"

    def create_properties(self, mesh, structuralmodel, geomodel):

        print('hello')
    