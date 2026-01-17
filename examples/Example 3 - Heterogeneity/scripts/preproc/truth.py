import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import geopandas as gpd
from matplotlib import gridspec
import sys
import loopflopy.utils as utils
import flopy

sys.path.append(r'C:\Users\00105295\Projects\Lab_tools\Geostats_tools')
from Krigger import optimized_kriging


def make_random_points(spatial, num_points=20, fname = '../data/random_points.xlsx'):
    # Create 20 random points
    def random_points_in_polygon(polygon, num_points):
        minx, miny, maxx, maxy = polygon.bounds
        points = []
        while len(points) < num_points:
            random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if polygon.contains(random_point):
                points.append(random_point)
        return points

    points = random_points_in_polygon(spatial.model_boundary_poly, num_points)
    x = [pt.x for pt in points]
    y = [pt.y for pt in points]
    val_TQ = [np.random.normal(0, 0.3) for _ in range(num_points)]  # 99% of values fall inside -1…1
    val_Kcok = [np.random.normal(0, 0.3) for _ in range(num_points)]  # 99% of values fall inside -1…1
    val_Kwlp = [np.random.normal(0, 0.3) for _ in range(num_points)]  # 99% of values fall inside -1…1
    val_Kwlw = [np.random.normal(0, 0.3) for _ in range(num_points)]  # 99% of values fall inside -1…1
    val_Kwlm = [np.random.normal(0, 0.3) for _ in range(num_points)]  # 99% of values fall inside -1…1

    df = pd.DataFrame({
                    'id': range(len(x)),
                    'x': x, 
                    'y': y, 
                    'val_TQ': val_TQ,
                    'val_Kcok': val_Kcok,
                    'val_Kwlp': val_Kwlp,
                    'val_Kwlw': val_Kwlw,
                    'val_Kwlm': val_Kwlm               
                    })

    df.to_excel(fname, index=False)

def plot_random_points(spatial):
     df = pd.read_excel('../data/random_points.xlsx')
     gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=spatial.epsg)
     
     fig, ax = plt.subplots() 
     ax.plot(gdf.geometry.x, gdf.geometry.y, 'o', ms = 2, label = 'Random Points')  # You can set color, linestyle, etc.
     x, y = spatial.model_boundary_poly.exterior.xy
     ax.plot(x, y, '-', ms = 2, lw = 1, color='black')
     ax.set_aspect('equal')
     ax.legend(loc ='lower left')
     plt.show()

def plot_fields(geomodel, mesh, field, title=None, levels=None, vmin=None, vmax=None, xy=None, xlim=None, ylim=None):

        fig, axes = plt.subplots(nrows=geomodel.nlg, ncols=geomodel.nls, figsize=(4*geomodel.nls, 3*geomodel.nlg))
        if title: fig.suptitle(title)
        for i in range(geomodel.nlg):
            for j in range(geomodel.nls):
                lay = i * geomodel.nls + j
                ax = axes[i, j]
                ax.set_title(f'{geomodel.units[i]} - Sublayer {j}')
                pmv = flopy.plot.PlotMapView(modelgrid=mesh.vgrid)
                array = field[lay]
                t = pmv.plot_array(array, ax = ax, vmin = vmin, vmax = vmax)
                cbar = fig.colorbar(t, ax=ax, shrink = 0.5)  
                #cg = pmv.contour_array(array, levels=levels, linewidths=0.8, colors="0.75")
                if xy:# Plot points
                    ax.plot(xy[0], xy[1], 'o', ms = 2, color = 'black')
                if xlim: ax.set_xlim(xlim) 
                if ylim: ax.set_ylim(ylim) 
                ax.tick_params(axis="both", labelsize=8)
        
        plt.tight_layout() 
        plt.subplots_adjust(top=0.92) 
        plt.show()

def plot_propfield(truth, mesh, geomodel, spatial, unit, sublay, property, xlim = None, ylim = None): # e.g xlim = [700000, 707500]

    array = getattr(truth, property)
    geolay = np.where(geomodel.units == unit)[0][0]
    lay = geomodel.nls * geolay + sublay  # zero based

    fig = plt.figure(figsize=(7,5))
    spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 0.05], wspace=0.2)

    ax = fig.add_subplot(spec[0], aspect="equal") #plt.subplot(1, 1, 1, aspect="auto")
    if xlim: ax.set_xlim(xlim) 
    if ylim: ax.set_ylim(ylim) 
    ax.set_title('Log %s - Unit %s, Sublayer %i' %(property, unit, sublay))
        
    pmv = flopy.plot.PlotMapView(ax = ax, modelgrid=mesh.vgrid)   
    p = pmv.plot_array(np.log10(array[lay]), alpha = 0.6, cmap = 'Spectral')#, norm = norm)    

    gdf = spatial.pilotpoint_gdf[spatial.pilotpoint_gdf.Unit == 'TQ']
    gdf.plot(ax=ax, markersize = 5, color = 'red', zorder=2)
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.id):
        ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 7, color = 'red', textcoords="offset points")

    spatial.obsbore_gdf.plot(ax=ax, markersize = 5, color = 'black', zorder=2)
    for x, y, label in zip(spatial.obsbore_gdf.geometry.x, spatial.obsbore_gdf.geometry.y, spatial.obsbore_gdf.id):
        ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 7, textcoords="offset points") 

    # Colorbar
    cbar_ax = fig.add_subplot(spec[1])
    cbar = fig.colorbar(p, cax=cbar_ax, shrink = 0.07)  # Center tick labels
    plt.savefig('../figures/%s_field_%s_sublay%i.png' %(property, unit, sublay), dpi = 300, bbox_inches='tight')  
    plt.show()

def plot_propfield_usg(truth, geomodel, spatial, property, x0, x1, y0, y1, **kwargs):
    
    array = getattr(truth, property)
    
    x0 = kwargs.get('x0', spatial.x0)
    y0 = kwargs.get('y0', spatial.y0)
    z0 = kwargs.get('z0', geomodel.z0)
    x1 = kwargs.get('x1', spatial.x1)
    y1 = kwargs.get('y1', spatial.y1)
    z1 = kwargs.get('z1', geomodel.z1)

    fig = plt.figure(figsize = (12,4))
    ax = plt.subplot(111)
    xsect = flopy.plot.PlotCrossSection(modelgrid=geomodel.vgrid , line={"line": [(x0, y0),(x1, y1)]}, geographic_coords=True)
    csa = xsect.plot_array(a = np.log10(array), cmap = 'Spectral', alpha=0.8)
    ax.set_xlabel('x (m)', size = 10)
    ax.set_ylabel('z (m)', size = 10)
    ax.set_ylim([z0, z1])
    linecollection = xsect.plot_grid(lw = 0.1, color = 'black') # Don't plot grid for reference
    
    cbar = plt.colorbar(csa,shrink = 1.0)
    plt.title(f"Log {property}, x0, x1, y0, y1 = {x0:.0f}, {x1:.0f}, {y0:.0f}, {y1:.0f}", size=8)
    plt.tight_layout()  
    plt.savefig('../figures/%s_field_usg_transect.png' %(property), dpi = 300, bbox_inches='tight')  
    plt.show()   

def plot_array_on_mesh(mesh, array, 
                       shapely_boundary = None,
                       vmin = None, 
                       vmax = None, 
                       levels = None, 
                       title = None, 
                       xlim = None, ylim = None,
                       xy = None,
                       df = None,
                       xsections = None,
                       xsection_names = None,
              ):

    if shapely_boundary:
        # Check which cells are inside the model boundary
        inside_boundary = []
        for icpl in range(mesh.ncpl):
            point = Point(mesh.xcyc[icpl])
            inside_boundary.append(shapely_boundary.contains(point))

        inside_boundary = np.array(inside_boundary)

        # Mask the array
        array = np.ma.masked_where(~inside_boundary, array)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot()
    
    pmv = flopy.plot.PlotMapView(modelgrid=mesh.vgrid)
    t = pmv.plot_array(array, vmin = vmin, vmax = vmax)
    cbar = plt.colorbar(t, shrink = 0.5)  

    if levels is not None:
        cg = pmv.contour_array(array.filled(np.nan), levels=levels, linewidths=0.8, colors="gray")
    
    if xy:  # Plot points
        ax.plot(xy[0], xy[1], 'o', ms = 2, color = 'black')

    if xlim: ax.set_xlim(xlim) 
    if ylim: ax.set_ylim(ylim) 
    if title: ax.set_title(title)

    if df is not None:
        ax.plot(df.X, df.Y, 'o', ms = 4, color = 'black', zorder=2)
        for x, y, label in zip(df.X, df.Y, df.ID):
            ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 8, 
                        color = 'black', textcoords="offset points")
            
    if xsections is not None:
        for i, xs in enumerate(xsections):
            x0, y0 = xs[0][0], xs[0][1]
            x1, y1 = xs[1][0], xs[1][1]
            ax.plot([x0,x1],[y0,y1], 'o-', ms = 2, lw = 2, color='black')
            name = xsection_names[i]
            ax.annotate(name, xy=(x0-1000, y0), xytext=(2, 2), size = 10, textcoords="offset points")

    plt.show()

def plot_isopachs(mesh, structuralmodel, 
                       shapely_boundary = None,
                       vmin = None, 
                       vmax = None, 
                       figsize = (12,8),
                       xlim = None, ylim = None,
                       xy = None,
                       dz = None,
                       title = None,
                       xsections = None,
                       xsection_names = None,
              ):

    if shapely_boundary:
        # Check which cells are inside the model boundary
        inside_boundary = []
        for icpl in range(mesh.ncpl):
            point = Point(mesh.xcyc[icpl])
            inside_boundary.append(shapely_boundary.contains(point))
        inside_boundary = np.array(inside_boundary)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout = True)
    if title : fig.suptitle(title, size=16)

    for i in range(len(structuralmodel.strat_names)-2):
        ax = axes.flat[i]
        ax.set_aspect('equal')

        unit = structuralmodel.strat_names[i+1]

        # filter dataframe for plotting
        df_filt = structuralmodel.data[structuralmodel.data['lithcode'] == unit]
        df = df_filt[['ID','X','Y', 'data_type']].drop_duplicates()

        # get unit thickness
        thickness = structuralmodel.surfaces[i] - structuralmodel.surfaces[i+1]
        thickness = np.ma.masked_where(~inside_boundary, thickness) # MAsk outside model boundary
        thickness[thickness < 0.] = np.nan
        max_thickness = float(np.nanmax(thickness))

        # make contour levels
        levels = np.arange(0, np.ceil(max_thickness / 10) * 10 + 10, dz[i])
    
        pmv = flopy.plot.PlotMapView(ax = ax, modelgrid=mesh.vgrid)
        t = pmv.plot_array(thickness, vmin = vmin, vmax = vmax)
        #cbar = plt.colorbar(t, shrink = 0.5) + 
        cbar = fig.colorbar(t, ax=ax, shrink=0.5, pad=0.01)

        if levels is not None:
            cg = pmv.contour_array(thickness.filled(np.nan), levels=levels, linewidths=0.8, colors="gray")
        
        if xy:  # Plot points
            ax.plot(xy[0], xy[1], 'o', ms = 2, color = 'black')

        if xlim: ax.set_xlim(xlim) 
        if ylim: ax.set_ylim(ylim) 
        ax.set_title(f'Thickness of {structuralmodel.strat_names[i+1]} in structural model',)

        if df is not None:
            ax.plot(df.X, df.Y, 'o', ms = 4, color = 'black', zorder=2)
            for x, y, label in zip(df.X, df.Y, df.ID):
                ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 8, 
                            color = 'black', textcoords="offset points")
                
        if xsections is not None:
            for i, xs in enumerate(xsections):
                x0, y0 = xs[0][0], xs[0][1]
                x1, y1 = xs[1][0], xs[1][1]
                ax.plot([x0,x1],[y0,y1], 'o-', ms = 2, lw = 2, color='black')
                name = xsection_names[i]
                ax.annotate(name, xy=(x0-1000, y0), xytext=(2, 2), size = 10, textcoords="offset points")

    plt.show()

def plot_xsections(structuralmodel, 
                       figsize = (12,8),
                       xlim = None, ylim = None,
                       dz = None,
                       xsections = None,
                       xsection_names = None,
                       title = None,
                       z0=None, z1=None,
                       nh=50, nz=50,
                        ):

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout = True)
    if title : fig.suptitle(title, size=16)

    for i in range(len(xsection_names)):
        ax = axes.flat[i]
        #ax.set_aspect('equal')

        x0 = xsections[i][0][0]
        y0 = xsections[i][0][1]
        x1 = xsections[i][1][0]
        y1 = xsections[i][1][1]


        x = np.linspace(x0, x1, nh)
        y = np.linspace(y0, y1, nh)        
        z = np.linspace(z0, z1, nz)

        X = np.tile(x, (len(z), 1)) 
        Y = np.tile(y, (len(z), 1)) 
        Z = np.tile(z[:, np.newaxis], (1, nh))  # Repeat z along columns (nh times)

        a = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
        V = structuralmodel.model.evaluate_model(a).reshape(np.shape(X))

        extent0 = 0
        extent1 = np.sqrt((x1-x0)**2 + (y1-y0)**2)

        csa = ax.imshow(np.ma.masked_where(V<0,V), origin = "lower", 
                         extent = [extent0, extent1, z0, z1], #[x0,x1,z0,z1], 
                         aspect = 'auto',
                         cmap = structuralmodel.cmap, norm = structuralmodel.norm, )
        
        labels = structuralmodel.strat_names[1:]
        ticks = [i for i in np.arange(0,len(labels))]
        boundaries = np.arange(-1,len(labels),1)+0.5

        cbar = plt.colorbar(csa,
                            boundaries = boundaries,
                            shrink = 0.5
                            )
        cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')    
        #plt.xticks(ticks = [], labels = [])
        if i == 2 or i == 3: ax.set_xlabel('Distance along transect (m)')
        if i == 0 or i == 2: ax.set_ylabel('Elev. (mAHD)')
        if i == 1 or i == 3: ax.set_yticks(ticks = [], labels = [])
        if i == 0 or i == 1: ax.set_xticks(ticks = [], labels = [])
        ax.set_title(xsection_names[i], size = 8)

        if xlim: ax.set_xlim(xlim) 
        if ylim: ax.set_ylim(ylim) 

    plt.show()

# This samples random points and uses kriging to make a stochastic field
class Truth1:    
    def __init__(self, **kwargs):      
        
        self.data_label = "DataBaseClass"
   
    def make_randomfield_3d(self, geomodel, mesh, spatial, scalarfield, df, units):

        def make_gdf(geomodel, mesh, spatial, scalarfield, df, unit):
            geolay = np.where(geomodel.units == unit)[0][0]

            gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df.x, df.y), crs=spatial.epsg)
            gdf['unit']      = unit
            gdf['prop_mean']  = 0.
            gdf['prop_var']   = 1.
            gdf['icpl']      = gdf['geometry'].apply(lambda point: mesh.gi.intersect(point)["cellids"][0])
            gdf['cell_xy']   = gdf['icpl'].apply(lambda icpl: (mesh.xcyc[icpl][0], mesh.xcyc[icpl][1]))
            gdf['unit']      = unit
            gdf['geolay']    = geolay
            gdf['lay']       = gdf['geolay'].apply(lambda geolay: geomodel.nls * geolay + 1)
            gdf['cell_disv'] = gdf.apply(lambda row: row['icpl'] + row['lay'] * mesh.ncpl, axis=1)  
            gdf['cell_disu'] = gdf.apply(lambda row: utils.disvcell_to_disucell(geomodel, row['cell_disv']), axis=1)  
            gdf['cell_z']  = gdf.apply(lambda row: geomodel.zc[row['lay'], row['icpl']], axis=1)
            gdf['sf_z']    = gdf.apply(lambda row: scalarfield[geolay][row['lay'], row['icpl']], axis=1)
            gdf = gdf[gdf['cell_disu'] != -1]
            
            return gdf

        gdf_list = []
        
        for unit in units:
            print(f"Processing unit: {unit}")
            gdf_unit = make_gdf(geomodel, mesh, spatial, scalarfield, df, unit)
            gdf_list.append(gdf_unit)
        
        # Combine all GeoDataFrames
        combined_gdf = pd.concat(gdf_list, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined_gdf, crs=gdf_list[0].crs)
        
        return combined_gdf

    # This is not purely random field = it takes 20 points and uses kriging
    def stochasticfieldgeneration(self, gdf, geomodel, mesh, anisotropy, corr_length, nugget, n_neighbors, rebuild_threshold, scalarfield):

        ## MAKE AN ARRAY OF X,Y,Z,VAL
        prop_layicpl = 999999 * np.ones((geomodel.nlay, geomodel.ncpl))

        for geolay, unit in enumerate(geomodel.units): # for each unit...
            print('\n###### ', unit, '#########\n')
            gdf_unit = gdf[gdf.unit == unit]
            pv = []

            for sublay in range(geomodel.nls): # for each sublayer in unit..
                print("Sublay = ", sublay)
                for icpl in range(geomodel.ncpl): # for each cell in plan...
                
                    lay = geolay * geomodel.nls + sublay # Zero based
                    disvcell = icpl + lay*geomodel.ncpl
                    x = mesh.xcyc[icpl][0]
                    y = mesh.xcyc[icpl][1]
                    z = geomodel.zc[lay, icpl] # scalarfield[geolay][lay, icpl] #

                    if disvcell in gdf_unit.cell_disv.values and geomodel.idomain[lay, icpl] == 1: # only include cells not pinched out
                        val = gdf_unit.loc[gdf_unit['cell_disv'] == disvcell, f'val_{unit}'].values[0]
            
                    else:
                        val = np.nan        
                         
                    pv.append([x, y, z, val])
            
            points_values_3d = np.array(pv)
            print('points_values_3d.shape ', points_values_3d.shape)
            #print('sill ', self.sills[geolay])
            
            n_initial_points = len(gdf_unit.id.values) # number of neighbours to use in kriging

            random_values = optimized_kriging(
                            points_values_3d, #x y z val USE LOG K
                            n_neighbors = n_neighbors, # determines how many neighbours
                            variogram_model="spherical", 
                            CL = corr_length,  # Correlation length
                            return_random = False, #False is kriging (deterministic), True is stochastoc
                            random_seed = None, 
                            n_initial_points=n_initial_points, # number of points to sample for initial variogram (assuming all points are nan)
                            unknown_value_flag=np.nan, 
                            anisotropy=anisotropy,
                            rebuild_threshold=rebuild_threshold) # how often to recreate KDTree e.g. every 0.1% of points rebuild KDTree'''

            points_values_3d[:,-1][np.isnan(points_values_3d[:,-1])] = random_values
            prop_values = points_values_3d[:,-1]

            # Normalise to -1 to 1
            prop_values = 2 * (prop_values - prop_values.min()) / (prop_values.max() - prop_values.min()) - 1
            print('prop_values ', prop_values.shape)

            prop_values = prop_values.reshape(geomodel.nls, geomodel.ncpl) # reshape into layers
            print('prop_values ', prop_values.shape)

            for sublay in range(geomodel.nls):
                lay = geolay * geomodel.nls + sublay # Zero based
                prop_layicpl[lay,:] = prop_values[sublay, :] # assign to the correct layer in the disv grid      

        prop_layicpl_ma = np.ma.masked_where(geomodel.idomain != 1, prop_layicpl)
        prop_disv = prop_layicpl_ma.flatten() # includes mask
        prop_disu = prop_layicpl_ma.compressed() # doesnt include mask

        setattr(self, f'log{property}_layicpl_ma', prop_layicpl_ma)
        setattr(self, f'log{property}_disv', prop_disv)
        setattr(self, f'log{property}_disu', prop_disu)
        setattr(self, f'{property}_layicpl_ma', 10**prop_layicpl_ma)
        setattr(self, f'{property}_disv', 10**prop_disv)
        setattr(self, f'{property}_disu', 10**prop_disu)
        print('len prop_layicpl_ma ', len(prop_layicpl_ma))
        print('prop_layicpl_ma ', prop_layicpl_ma[0].shape)

        return prop_layicpl_ma
    
    def make_prop_fields(self, geomodel, prop_df):

        kh = np.empty_like(self.field)
        kv = np.empty_like(self.field)
        ss = np.empty_like(self.field)
        sy = np.empty_like(self.field)

        for i, unit in enumerate(geomodel.units):
            print('\n###### ', unit, '#########\n')
            df = prop_df[prop_df['unit'] == unit]
            for j in range(geomodel.nls):
                lay = i * geomodel.nls + j
                print(f'Unit {i}: {unit}')

                # kh
                kh_logmean = np.log10(df.kh_mean.values[0])
                kh_logstd = df.kh_logstd.values[0]
                kh[lay] = kh_logmean + self.field[lay] *  kh_logstd

                # kv
                kv_logmean = np.log10(df.kv_mean.values[0])
                kv_logstd = df.kv_logstd.values[0]
                kv[lay] = kv_logmean + self.field[lay] * kv_logstd

                # ss
                ss_logmean = np.log10(df.ss_mean.values[0])
                ss_logstd = df.ss_logstd.values[0]
                ss[lay] = ss_logmean + self.field[lay] * ss_logstd

                # sy
                sy_logmean = np.log10(df.sy_mean.values[0])
                sy_logstd = df.sy_logstd.values[0]
                sy[lay] = sy_logmean + self.field[lay] * sy_logstd

        self.kh = np.ma.masked_where(geomodel.idomain != 1, 10**kh)
        self.kv = np.ma.masked_where(geomodel.idomain != 1, 10**kv)
        self.ss = np.ma.masked_where(geomodel.idomain != 1, 10**ss)
        self.sy = np.ma.masked_where(geomodel.idomain != 1, 10**sy)

    def print_prop_stats(self):
        import numpy as np

        print("\n### Property statistics ###\n")

        header = (
            f"{'Property':<8} {'Mean':>12} {'Min':>12} {'Max':>12} "
            f"{'LogMean':>12} {'LogMin':>12} {'LogMax':>12}"
        )
        print(header)
        print("-" * len(header))

        for prop in ['kh', 'kv', 'ss', 'sy']:
            arr = getattr(self, prop)
            log_arr = np.log10(arr)

            print(
                f"{prop:<8} "
                f"{arr.mean():12.4e} {arr.min():12.4e} {arr.max():12.4e} "
                f"{log_arr.mean():12.4e} {log_arr.min():12.4e} {log_arr.max():12.4e}"
            )

# Totally random field - does not use any points
class Truth2:    
    def __init__(self, **kwargs):      
        
        self.data_label = "DataBaseClass"

    def stochasticfieldgeneration(self, geomodel, mesh, anisotropy, corr_length, nugget, n_neighbours, n_initial_points, rebuild_threshold, scalarfield):

        ## MAKE AN ARRAY OF X,Y,Z,VAL
        prop_layicpl = 999999 * np.ones((geomodel.nlay, geomodel.ncpl))

        for geolay, unit in enumerate(geomodel.units): # for each unit...
            print('###### ', unit, '#########\n')
            pv = []

            for sublay in range(geomodel.nls): # for each sublayer in unit..
                print("Sublay = ", sublay)
                for icpl in range(geomodel.ncpl): # for each cell in plan...
                
                    lay = geolay * geomodel.nls + sublay # Zero based
                    x = mesh.xcyc[icpl][0]
                    y = mesh.xcyc[icpl][1]
                    z = geomodel.zc[lay, icpl] # scalarfield[geolay][lay, icpl] #      
                    pv.append([x, y, z, np.nan])
            
            points_values_3d = np.array(pv)
            print('points_values_3d.shape ', points_values_3d.shape)
            
            random_values = optimized_kriging(
                            points_values_3d, #x y z val USE LOG K
                            n_neighbors = n_neighbours, # determines how many neighbours
                            variogram_model="spherical", 
                            CL = corr_length,  # Correlation length
                            return_random = True, #False is kriging (deterministic), True is stochastoc
                            random_seed = None, 
                            n_initial_points = n_initial_points, # number of points to sample for initial variogram (assuming all points are nan)
                            unknown_value_flag=np.nan, 
                            anisotropy=anisotropy,
                            rebuild_threshold=rebuild_threshold) # how often to recreate KDTree e.g. every 0.1% of points rebuild KDTree'''

            points_values_3d[:,-1][np.isnan(points_values_3d[:,-1])] = random_values
            prop_values = points_values_3d[:,-1]

            # Normalise to -1 to 1
            prop_values = 2 * (prop_values - prop_values.min()) / (prop_values.max() - prop_values.min()) - 1
            print('prop_values ', prop_values.shape)

            prop_values = prop_values.reshape(geomodel.nls, geomodel.ncpl) # reshape into layers
            print('prop_values ', prop_values.shape)

            for sublay in range(geomodel.nls):
                lay = geolay * geomodel.nls + sublay # Zero based
                prop_layicpl[lay,:] = prop_values[sublay, :] # assign to the correct layer in the disv grid
        
        prop_layicpl_ma = np.ma.masked_where(geomodel.idomain != 1, prop_layicpl)
        prop_disv = prop_layicpl_ma.flatten() # includes mask
        prop_disu = prop_layicpl_ma.compressed() # doesnt include mask


        return prop_layicpl_ma
    
    def make_prop_fields(self, geomodel, prop_df):

        kh = np.empty_like(self.field)
        kv = np.empty_like(self.field)
        ss = np.empty_like(self.field)
        sy = np.empty_like(self.field)

        for i, unit in enumerate(geomodel.units):
            print('\n###### ', unit, '#########\n')
            df = prop_df[prop_df['unit'] == unit]
            for j in range(geomodel.nls):
                lay = i * geomodel.nls + j
                print(f'Unit {i}: {unit}')

                # kh
                kh_logmean = np.log10(df.kh_mean.values[0])
                kh_logstd = df.kh_logstd.values[0]
                kh[lay] = kh_logmean + self.field[lay] *  kh_logstd

                # kv
                kv_logmean = np.log10(df.kv_mean.values[0])
                kv_logstd = df.kv_logstd.values[0]
                kv[lay] = kv_logmean + self.field[lay] * kv_logstd

                # ss
                ss_logmean = np.log10(df.ss_mean.values[0])
                ss_logstd = df.ss_logstd.values[0]
                ss[lay] = ss_logmean + self.field[lay] * ss_logstd

                # sy
                sy_logmean = np.log10(df.sy_mean.values[0])
                sy_logstd = df.sy_logstd.values[0]
                sy[lay] = sy_logmean + self.field[lay] * sy_logstd

        self.kh = np.ma.masked_where(geomodel.idomain != 1, 10**kh)
        self.kv = np.ma.masked_where(geomodel.idomain != 1, 10**kv)
        self.ss = np.ma.masked_where(geomodel.idomain != 1, 10**ss)
        self.sy = np.ma.masked_where(geomodel.idomain != 1, 10**sy)

    def print_prop_stats(self):
        import numpy as np

        print("\n### Property statistics ###\n")

        header = (
            f"{'Property':<8} {'Mean':>12} {'Min':>12} {'Max':>12} "
            f"{'LogMean':>12} {'LogMin':>12} {'LogMax':>12}"
        )
        print(header)
        print("-" * len(header))

        for prop in ['kh', 'kv', 'ss', 'sy']:
            arr = getattr(self, prop)
            log_arr = np.log10(arr)

            print(
                f"{prop:<8} "
                f"{arr.mean():12.4e} {arr.min():12.4e} {arr.max():12.4e} "
                f"{log_arr.mean():12.4e} {log_arr.min():12.4e} {log_arr.max():12.4e}"
            )
