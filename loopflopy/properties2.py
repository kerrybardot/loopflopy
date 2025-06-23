import geopandas as gpd
import flopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import sys
import loopflopy.utils as utils

sys.path.append(r'C:\Users\00105295\Projects\Lab_tools\Geostats_tools')
from Krigger import optimized_kriging

class Properties:    
    def __init__(self, **kwargs):      
        
        self.data_label = "DataBaseClass"
   
    def make_stochastic_gdf(self, geomodel, mesh, spatial, scalarfield, df, unit):

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Easting, df.Northing), crs=spatial.epsg)
        gdf['prop_mean']  = 0.
        gdf['prop_var']   = 1.
        gdf['icpl']      = gdf['geometry'].apply(lambda point: mesh.gi.intersect(point)["cellids"][0])
        gdf['cell_xy']   = gdf['icpl'].apply(lambda icpl: (mesh.xcyc[icpl][0], mesh.xcyc[icpl][1]))
        gdf['unit']      = unit
        gdf['geolay']    = gdf['unit'].apply(lambda unit: np.where(geomodel.units == unit)[0][0]) # Find the index of the strat name in the list
        gdf['lay']       = gdf['geolay'].apply(lambda geolay: geomodel.nls * geolay + 1) # Bottom layer for 2 sublayers, middle layer for 3 sublayers  ##########################                                                          
        gdf['cell_disv'] = gdf.apply(lambda row: row['icpl'] + row['lay'] * mesh.ncpl, axis=1)  
        gdf['cell_disu'] = gdf.apply(lambda row: utils.disvcell_to_disucell(geomodel, row['cell_disv']), axis=1)  
        gdf['cell_z']  = gdf.apply(lambda row: geomodel.zc[row['lay'], row['icpl']], axis=1)
        gdf['sf_z']    = gdf.apply(lambda row: scalarfield[row['lay'], row['icpl']], axis=1)
        gdf = gdf[gdf['cell_disu'] != -1] # delete pilot points where layer is pinched out

        return gdf
    
    def stochasticfieldgeneration(self, 
                unit,
                gdf,
                geomodel, 
                mesh, 
                property = 'prop',
                anisotropy = (1., 1., 1.), 
                return_random=True, # True makes it random with 0 mean and 1 variance, False makes it deterministic
                CL = 1000., 
                nugget = 0.05, 
                rebuild_threshold = 0.1,):

        ## MAKE AN ARRAY OF X,Y,Z,VAL
        prop_layicpl = 999999 * np.ones((geomodel.nlay, geomodel.ncpl))

        for geolay, unit in enumerate(geomodel.units): # for each unit...
            print('###### ', unit, '#########\n')
            
            pv = []

            for sublay in range(geomodel.nls): # for each sublayer in unit..
                for icpl in range(geomodel.ncpl): # for each cell in plan...
                
                    lay = geolay * geomodel.nls + sublay # Zero based
                    disvcell = icpl + lay*geomodel.ncpl
                    x = mesh.xcyc[icpl][0]
                    y = mesh.xcyc[icpl][1]
                    z = geomodel.zc[lay, icpl]

                    if disvcell in gdf.cell_disv.values and geomodel.idomain[lay, icpl] == 1: # only include cells not pinched out
                        id = gdf.loc[gdf['cell_disv'] == disvcell, 'ID'].values[0]
                        print(id, disvcell)
                        val = 0.
                        #val = np.log10(gdf.loc[gdf['cell_disv'] == disvcell, property].values)[0]
                        print(id, unit, 'disv_cell', disvcell, 'val', val)
            
                    else:
                        val = np.nan        
                            
                    pv.append([x, y, z, val])
            
            points_values_3d = np.array(pv)
            print('points_values_3d.shape ', points_values_3d.shape)
            print('sill ', self.sills[geolay])
            
            n_neighbors = len(gdf.ID.values) # number of neighbours to use in kriging
            print(n_neighbors)
            random_values = optimized_kriging(points_values_3d, #x y z val USE LOG K
                            n_neighbors = n_neighbors, # determines how many neighbours
                            variogram_model="spherical", 
                            CL = CL,  # Correlation length
                            sill = self.sills[geolay],  # maximum variance - replace with spreadsheet
                            nugget = nugget, # value at 0 distance
                            return_random = True, #False is kriging (deterministic), True is stochastoc
                            random_seed = None, 
                            initial_points=20, # number of points to sample for initial variogram (assuming all points are nan)
                            unknown_value_flag=np.nan, 
                            anisotropy=anisotropy,
                            rebuild_threshold=rebuild_threshold) # how often to recreate KDTree e.g. every 0.1% of points rebuild KDTree'''

            points_values_3d[:,-1][np.isnan(points_values_3d[:,-1])] = random_values
            prop_values = points_values_3d[:,-1]
            print('prop_values ', prop_values.shape)

            prop_values = prop_values.reshape(geomodel.nls, geomodel.ncpl) # reshape into layers
            print('prop_values ', prop_values.shape)

            for sublay in range(geomodel.nls):
                lay = geolay * geomodel.nls + sublay # Zero based
                prop_layicpl[lay,:] = prop_values[sublay, :] # assign to the correct layer in the disv grid
        
        prop_layicpl_ma = np.ma.masked_where(geomodel.idomain != 1, prop_layicpl)
        prop_disv = prop_layicpl_ma.flatten()
        prop_disu = prop_layicpl_ma.compressed()

        setattr(self, f'log{property}_layicpl_ma', prop_layicpl_ma)
        setattr(self, f'log{property}_disv', prop_disv)
        setattr(self, f'log{property}_disu', prop_disu)
        setattr(self, f'{property}_layicpl_ma', 10**prop_layicpl_ma)
        setattr(self, f'{property}_disv', 10**prop_disv)
        setattr(self, f'{property}_disu', 10**prop_disu)
    
    '''
    
    '''
    
    def kriging(self, 
                geomodel, 
                mesh, 
                property = 'kh',
                anisotropy = (1., 1., 1.), 
                return_random=False, # True makes it random with 0 mean and 1 variance, False makes it deterministic
                CL = 1000., 
                nugget = 0.05, 
                rebuild_threshold = 0.1):

        ## MAKE AN ARRAY OF X,Y,Z,VAL
        prop_layicpl = 999999 * np.ones((geomodel.nlay, geomodel.ncpl))


        for geolay, unit in enumerate(geomodel.units): # for each unit...
            print('###### ', unit, '#########\n')
            gdf = self.gdf[self.gdf.Unit == unit]
            pv = []

            for sublay in range(geomodel.nls): # for each sublayer in unit..
                for icpl in range(geomodel.ncpl): # for each cell in plan...
                
                    lay = geolay * geomodel.nls + sublay # Zero based
                    disvcell = icpl + lay*geomodel.ncpl
                    
                    x = mesh.xcyc[icpl][0]
                    y = mesh.xcyc[icpl][1]
                    z = geomodel.zc[lay, icpl]

                    if disvcell in gdf.cell_disv.values and geomodel.idomain[lay, icpl] == 1: # only include cells not pinched out
                        id = gdf.loc[gdf['cell_disv'] == disvcell, 'ID'].values[0]
                        val = np.log10(gdf.loc[gdf['cell_disv'] == disvcell, property].values)[0]
                        print(id, unit, 'disv_cell', disvcell, 'val', val)
            
                    else:
                        val = np.nan        
                            
                    pv.append([x, y, z, val])
            
            points_values_3d = np.array(pv)
            print('points_values_3d.shape ', points_values_3d.shape)
            print('sill ', self.sills[geolay])
            
            ## RUN KRIGING
            n_neighbors = len(gdf.ID.values) # number of neighbours to use in kriging
            print(n_neighbors)
            random_values = optimized_kriging(points_values_3d, #x y z val USE LOG K
                            n_neighbors = n_neighbors, # determines how many neighbours
                            variogram_model="spherical", 
                            CL = CL,  # Correlation length
                            sill = self.sills[geolay],  # maximum variance - replace with spreadsheet
                            nugget = nugget, # value at 0 distance
                            return_random = False, #False is kriging (deterministic), True is stochastoc
                            random_seed = None, 
                            #n_initial_points=10, # doesnt use if existing points, number of points to sample for initial variogram (assuming all points are nan)
                            unknown_value_flag=np.nan, 
                            anisotropy=anisotropy,
                            rebuild_threshold=rebuild_threshold) # how often to recreate KDTree e.g. every 0.1% of points rebuild KDTree'''

            points_values_3d[:,-1][np.isnan(points_values_3d[:,-1])] = random_values
            prop_values = points_values_3d[:,-1]
            print('prop_values ', prop_values.shape)

            prop_values = prop_values.reshape(geomodel.nls, geomodel.ncpl) # reshape into layers
            print('prop_values ', prop_values.shape)

            for sublay in range(geomodel.nls):
                lay = geolay * geomodel.nls + sublay # Zero based
                prop_layicpl[lay,:] = prop_values[sublay, :] # assign to the correct layer in the disv grid
        
        prop_layicpl_ma = np.ma.masked_where(geomodel.idomain != 1, prop_layicpl)
        prop_disv = prop_layicpl_ma.flatten()
        prop_disu = prop_layicpl_ma.compressed()

        setattr(self, f'log{property}_layicpl_ma', prop_layicpl_ma)
        setattr(self, f'log{property}_disv', prop_disv)
        setattr(self, f'log{property}_disu', prop_disu)
        setattr(self, f'{property}_layicpl_ma', 10**prop_layicpl_ma)
        setattr(self, f'{property}_disv', 10**prop_disv)
        setattr(self, f'{property}_disu', 10**prop_disu)
        

    def plot_propfield(self, mesh, spatial, lay, unit, property = 'kh', xlim = None, ylim = None): # e.g xlim = [700000, 707500]
    
        array = getattr(self, f'log{property}_layicpl_ma')

        fig = plt.figure(figsize=(7,5))
        spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 0.05], wspace=0.2)

        ax = fig.add_subplot(spec[0], aspect="equal") #plt.subplot(1, 1, 1, aspect="auto")
        if xlim: ax.set_xlim(xlim) 
        if ylim: ax.set_ylim(ylim) 
        ax.set_title('Log %s - Unit %s' %(property, unit))
            
        pmv = flopy.plot.PlotMapView(ax = ax, modelgrid=mesh.vgrid)   
        #mask = mesh.idomain == 0
        #ma = np.ma.masked_where(mask, k_values)
        p = pmv.plot_array(array[lay], alpha = 0.6, cmap = 'Spectral')#, norm = norm)    

        gdf = spatial.pilotpoint_gdf[spatial.pilotpoint_gdf.Unit == 'TQ']
        gdf.plot(ax=ax, markersize = 5, color = 'red', zorder=2)
        for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.ID):
            ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 7, color = 'red', textcoords="offset points")

        spatial.obsbore_gdf.plot(ax=ax, markersize = 5, color = 'black', zorder=2)
        for x, y, label in zip(spatial.obsbore_gdf.geometry.x, spatial.obsbore_gdf.geometry.y, spatial.obsbore_gdf.ID):
            ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 7, textcoords="offset points") 
    
        # Colorbar
        cbar_ax = fig.add_subplot(spec[1])
        cbar = fig.colorbar(p, cax=cbar_ax, shrink = 0.1)  # Center tick labels
        plt.savefig('../figures/%s_field_%s_lay%i.png' %(property, unit, lay), dpi = 300, bbox_inches='tight')  
        plt.show()

    def plot_propfield_usg(self, geomodel, spatial, x0, x1, y0, y1, property = 'kh', **kwargs):
        
        array = getattr(self, f'log{property}_layicpl_ma')
        
        x0 = kwargs.get('x0', spatial.x0)
        y0 = kwargs.get('y0', spatial.y0)
        z0 = kwargs.get('z0', geomodel.z0)
        x1 = kwargs.get('x1', spatial.x1)
        y1 = kwargs.get('y1', spatial.y1)
        z1 = kwargs.get('z1', geomodel.z1)

        fig = plt.figure(figsize = (12,4))
        ax = plt.subplot(111)
        xsect = flopy.plot.PlotCrossSection(modelgrid=geomodel.vgrid , line={"line": [(x0, y0),(x1, y1)]}, geographic_coords=True)
        csa = xsect.plot_array(a = array, cmap = 'Spectral', alpha=0.8)
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('z (m)', size = 10)
        ax.set_ylim([z0, z1])
        linecollection = xsect.plot_grid(lw = 0.1, color = 'black') # Don't plot grid for reference
        
        cbar = plt.colorbar(csa,shrink = 1.0)
        plt.title(f"Log {property}, x0, x1, y0, y1 = {x0:.0f}, {x1:.0f}, {y0:.0f}, {y1:.0f}", size=8)
        plt.tight_layout()  
        plt.savefig('../figures/%s_field_usg_transect.png' %(property), dpi = 300, bbox_inches='tight')  
        plt.show()   
