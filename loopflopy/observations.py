import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import loopflopy.utils as utils

class Observations:
    """
    A class for managing observation data in groundwater flow models.
    
    This class handles observation well data for MODFLOW 6 simulations, including
    coordinate transformation, cell location, and preparation of observation
    records for the MODFLOW 6 OBS package. It works with GeoPandas DataFrames
    containing observation well locations and properties.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing observation well data with the following required columns:
        - 'x': X-coordinates of observation wells
        - 'y': Y-coordinates of observation wells  
        - 'z': Z-coordinates (elevations) of observation wells
        - 'id': Unique identifier for each observation well
        - 'cell_disu': DISU cell index for each well location
    **kwargs
        Additional keyword arguments to set as instance attributes.
    
    Attributes
    ----------
    observations_label : str
        Label identifier for the observations (default: "ObservationsBaseClass").
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing observation well data.
    obs_rec : list
        Record array formatted for MODFLOW 6 OBS package input.
        Created by calling make_recarray() method.
    
    Notes
    -----
    The class is designed to work with MODFLOW 6's unstructured DISU grids
    and requires that observation well locations have already been mapped
    to model cells (cell_disu column).
    
    The observation records are formatted as:
    [obs_id, 'head', cell_number_1based]
    
    This follows MODFLOW 6 OBS package conventions where:
    - obs_id: Unique identifier for the observation
    - 'head': Observation type (hydraulic head)
    - cell_number: 1-based cell index for MODFLOW input
    
    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from shapely.geometry import Point
    >>>
    >>> # Create observation well data
    >>> data = {
    ...     'id': ['MW1', 'MW2', 'MW3'],
    ...     'x': [700000, 705000, 710000],
    ...     'y': [6200000, 6205000, 6210000],
    ...     'z': [50, 45, 40],
    ...     'cell_disu': [100, 250, 400]
    ... }
    >>> geometry = [Point(x, y) for x, y in zip(data['x'], data['y'])]
    >>> gdf = gpd.GeoDataFrame(data, geometry=geometry)
    >>>
    >>> # Create observations object
    >>> obs = Observations(gdf, project_name='test_model')
    >>>
    >>> # Generate MODFLOW observation records
    >>> obs.make_recarray()
    >>> print(obs.obs_rec)
    [['MW1', 'head', 101], ['MW2', 'head', 251], ['MW3', 'head', 401]]
    
    See Also
    --------
    Mesh.locate_special_cells : For identifying observation well cells in mesh
    flopy.mf6.ModflowGwfobs : MODFLOW 6 OBS package for using observation records
    """
    def __init__(self, gdf, **kwargs):
        """
        Initialize Observations object with observation well data.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with observation well locations and properties.
        **kwargs
            Additional attributes to set on the instance.
        
        Notes
        -----
        Sets default observations_label and stores the GeoDataFrame.
        Additional keyword arguments are dynamically added as instance attributes.
        """

        self.observations_label = "ObservationsBaseClass"
        self.gdf = gdf
        
        for key, value in kwargs.items():
            setattr(self, key, value)       
    
    def make_recarray(self):
        """
        Create MODFLOW 6 observation record array from GeoDataFrame.
        
        Converts the observation well data into the format required by
        MODFLOW 6's OBS package for hydraulic head observations.
        
        Notes
        -----
        The method:
        1. Extracts coordinates (x, y, z) from the GeoDataFrame
        2. Gets observation IDs and corresponding DISU cell indices
        3. Creates observation records in MODFLOW 6 format
        4. Converts cell indices from 0-based (Python) to 1-based (MODFLOW)
        
        Each observation record contains:
        - obs_id: Unique identifier from 'id' column
        - 'head': Observation type (hydraulic head)
        - cell_number: 1-based DISU cell index
        
        Raises
        ------
        AttributeError
            If required columns ('id', 'x', 'y', 'z', 'cell_disu') are missing
            from the GeoDataFrame.
        """

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
        