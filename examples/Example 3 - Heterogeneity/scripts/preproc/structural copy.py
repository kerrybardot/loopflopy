import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
from LoopStructural.utils import strikedip2vector as strike_dip_vector
import geopandas as gpd
from loopflopy.mesh_routines import resample_linestring
import rasterio
from rasterio.warp import transform

def find_elevation(x_coord, y_coord, spatial, dem_tif):
    # create point tuple
    pt = (x_coord, y_coord)

    with rasterio.open(dem_tif) as src:
        # ensure point is in same CRS as raster
        if src.crs.to_string() != spatial.crs: #"EPSG:your_crs":
            x, y = transform(spatial.crs, src.crs, [pt[0]], [pt[1]])
            pt = (x[0], y[0])

        # sample returns a generator
        value = list(src.sample([pt]))[0][0]

    return(value)

def prepare_strat_column(structuralmodel):
    
    strat = pd.read_excel(structuralmodel.geodata_fname, sheet_name = structuralmodel.strat_sheetname)
    strat_names = strat.unit.tolist()
    lithids = strat.lithid.tolist()
    print(strat_names)
    vals = strat.val.tolist()
    nlg = len(strat_names) - 1 # number of geological layers
    sequences = strat.sequence.tolist()
    sequence = list(dict.fromkeys(sequences)) # Preserves order and removes duplicates
    print(sequence)

    # Make bespoke colormap
    stratcolors = []
    for i in range(len(strat)):
        R = strat.R.loc[i].item() / 255
        G = strat.G.loc[i].item() / 255
        B = strat.B.loc[i].item() / 255
        stratcolors.append([round(R, 2), round(G, 2), round(B, 2)])
    
    import matplotlib.colors
    norm=plt.Normalize(min(lithids),max(lithids))
    tuples = list(zip(map(norm,lithids), stratcolors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    ##########################
    stratigraphic_column = {}
    for i in range(len(sequence)):
        stratigraphic_column[sequence[i]] = {}
    for i in range(len(sequences)):
        if i == 0 or sequences[i] != sequences[i-1]:
            mx = np.inf
        else:
            mx = vals[i-1]

        if i == (len(sequences) - 1): # LAST UNIT! #i != (len(sequences) - 1) and sequences[i] != sequences[i+1] and i != 1:
            mn = -np.inf
        else:
            mn = vals[i]
        if i == 0: 
            mn = vals[i] #work around for the ground
        stratigraphic_column[sequences[i]][strat_names[i]] = {'min': mn, 'max': mx, 'id': lithids[i], 'color': stratcolors[i]}
    ###########################    
           
    structuralmodel.strat = strat
    structuralmodel.strat_col = stratigraphic_column
    structuralmodel.strat_names = strat_names
    structuralmodel.cmap = cmap
    structuralmodel.norm = norm
    structuralmodel.sequence = sequence
    structuralmodel.lithids = lithids
    structuralmodel.sequences = sequences
    structuralmodel.vals = vals
    
def prepare_geodata(structuralmodel, spatial, dem_tif, control = False, truth = False,Fault = True, extent = None):  

    strat = structuralmodel.strat
    df = pd.read_excel(structuralmodel.geodata_fname, sheet_name=structuralmodel.data_sheetname)

# ---------- Prepare borehole data ----------------
    data_list = df.values.tolist()  # Turn data into a list of lists
    formatted_data = []
    for i in range(len(data_list)): #iterate for each row
        
        data_type = data_list[i][3]  
        
        #-----------RAW DATA-----------------------
        if data_type == 'Raw': ### Z VALUES: Ground (mAHD), Else: (mBGL)
            
            boreid = data_list[i][0]
            x, y = data_list[i][1], data_list[i][2]
            #groundlevel = data_list[i][5]  
            groundlevel = find_elevation(x, y, spatial, dem_tif) ## NEW!!!!!####

            # Add ground level to dataframe
            formatted_data.append([boreid, x, y, groundlevel, 0, 'Ground', 'Ground', 0, 0, 1, data_type]) 
            count = 1  # Add data row for each lithology
            for j in range(6,df.shape[1]-1): #iterate through each formation 
                if isinstance(data_list[i][j], numbers.Number) == True:  # Add lithology  
                    bottom    = groundlevel - float(data_list[i][j])  # Ground surface - formation bottom (mbgl)
                    val       = strat.val[count]                   # designated isovalue
                    unit      = strat.unit[count]                  # unit 
                    feature   = strat.sequence[count]              # sequence
                    gx, gy, gz = 0,0,1                             # normal vector to surface (flat)            
                    formatted_data.append([boreid, x, y, bottom, val, unit, feature, gx, gy, gz, data_type])    
                    current_bottom = np.copy(bottom)    
                count+=1
        
            #-----------CONTROL POINT-----------------------
        if control:
            if data_type == 'Control': ### Z VALUES: mAHD
                boreid = data_list[i][0]
                x, y = data_list[i][1], data_list[i][2]
                #groundlevel = data_list[i][5] 
                groundlevel = find_elevation(x, y, spatial, dem_tif) ## NEW!!!!!####

                count = 1  # Add data row for each lithology
                for j in range(6,df.shape[1]-1): #iterate through each formation 
                    if isinstance(data_list[i][j], numbers.Number) == True:  # Add lithology  
                        Z         = groundlevel - float(data_list[i][j])   # elevation (mAHD)
                        val       = strat.val[count]                       # designated isovalue
                        unit      = strat.unit[count]                      # unit 
                        feature   = strat.sequence[count]                  # sequence
                        if unit == 'Ground':
                            gx, gy, gz = 0,0,1                             # normal vector to surface (flat) 
                        else:
                            gx, gy, gz = np.nan, np.nan, np.nan
                        formatted_data.append([boreid, x, y, Z, val, unit, feature, gx, gy, gz, data_type])      
                    count+=1

                    #-----------TRUTH-----------------------
        if truth:
            if data_type == 'Truth': ### Z VALUES: mAHD
                boreid = data_list[i][0]
                x, y = data_list[i][1], data_list[i][2]
                groundlevel = find_elevation(x, y, spatial, dem_tif) # interpolate ground level from DEM

                count = 1  # Add data row for each lithology
                for j in range(6,df.shape[1]-1): #iterate through each formation 
                    if isinstance(data_list[i][j], numbers.Number) == True:  # Add lithology  
                        Z         = groundlevel - float(data_list[i][j])   # elevation (mAHD)
                        val       = strat.val[count]                       # designated isovalue
                        unit      = strat.unit[count]                      # unit 
                        feature   = strat.sequence[count]                  # sequence
                        if unit == 'Ground':
                            gx, gy, gz = 0,0,1                             # normal vector to surface (flat) 
                        else:
                            gx, gy, gz = np.nan, np.nan, np.nan
                        formatted_data.append([boreid, x, y, Z, val, unit, feature, gx, gy, gz, data_type])      
                    count+=1
                
    data = pd.DataFrame(formatted_data)
    data.columns =['ID','X','Y','Z','val','lithcode','feature_name', 'gx', 'gy', 'gz','data_type']
    
    # ---------- Prepare fault details ----------------   
    if Fault:

        fx, fy = [], []
        for point in spatial.faults_gdf.geometry[0].coords:
            x,y = point[0], point[1]
            fx.append((x))
            fy.append((y))
        fz = [-500] # making a plane

        from LoopStructural.utils import strikedip2vector # array of arrays
        dip = 90 # Vertical fault
        fault_rows= []

        for v in range(len(fz)):# vertical points 
            z = fz[v]
            for n in range(len(fx)-1): # fault segments = number of fault points - 1

                if fx[n+1] - fx[n] == 0: # fault due N-S
                    strike = 0
                    x = fx[n]
                    if fy[n+1] > fy[n]: # moving north
                        y = fy[n] + (fy[n+1] - fy[n]) / 2
                        nx, ny, nz = strikedip2vector([strike], [dip])[0]
                        fault_rows.append([x, y, z, nx, ny, nz])

                    else: # moving south
                        y = fy[n] - (fy[n+1] - fy[n]) / 2
                        nx, ny, nz = strikedip2vector([strike], [dip])[0]
                        fault_rows.append([x, y, z, nx, ny, nz])
                else:
                    grad = (fy[n+1] - fy[n]) / (fx[n+1] - fx[n]) # gradient of segment
                    x = fx[n] + (fx[n+1] - fx[n])/2 # midpoint along fault segment
                    y = fy[n] + (fy[n+1] - fy[n])/2 # midpoint along fault segment
                    strike = np.rad2deg(np.arctan((fx[n+1] - fx[n]) / abs(fy[n+1] - fy[n])))
                    #strike = 90 - np.rad2deg(np.arctan(abs(fy[n+1] - fy[n]) / (fx[n+1] - fx[n])))
                    nx, ny, nz = strikedip2vector([strike], [dip])[0]
                    fault_rows.append([x, y, z, nx, ny, nz])
        
        for i in fault_rows:
            df_new_row = pd.DataFrame.from_records(
                    {
                        "X": [i[0]],
                        "Y": [i[1]],
                        "Z": [i[2]],
                        "val": [0.0],
                        "feature_name": ["Fault"],
                        "nx": [i[3]],
                        "ny": [i[4]],
                        "nz": [i[5]],
                        "ID" : ["Fault_cloud"],
                        "data_type" : ["Fault"]
                    }
                )
            data = pd.concat([data, df_new_row], ignore_index=True)      

    
    structuralmodel.data = data

def create_structuralmodel(structuralmodel, nelements = 1e4, Fault = True):
    
    origin  = (structuralmodel.x0, structuralmodel.y0, structuralmodel.z0)
    maximum = (structuralmodel.x1, structuralmodel.y1, structuralmodel.z1)

    import LoopStructural
    from LoopStructural import GeologicalModel
    print(LoopStructural.__version__)

    model = GeologicalModel(origin, maximum)
    model.set_model_data(structuralmodel.data)  
    
    Ground     = model.create_and_add_foliation("Ground", nelements=nelements)
    Ground_UC  = model.add_unconformity(Ground, structuralmodel.strat[structuralmodel.strat.unit == 'Ground'].val.iloc[0]) 
    TQ         = model.create_and_add_foliation("TQ", nelements=nelements)
    TQ_UC      = model.add_unconformity(TQ, structuralmodel.strat[structuralmodel.strat.unit == 'TQ'].val.iloc[0]) 
    
    if Fault:
        print('Fault included!')
        Fault = model.create_and_add_fault('Fault', displacement = 400,)
        structuralmodel.faults = ['Fault']
        
    Kcok    = model.create_and_add_foliation("Kcok", nelements=nelements, buffer=0.1)
    Kcok_UC = model.add_unconformity(Kcok, structuralmodel.strat[structuralmodel.strat.unit == 'Kcok'].val.iloc[0]) 
    Leed    = model.create_and_add_foliation("Leed", nelements=nelements, buffer=0.1)    
    model.set_stratigraphic_column(structuralmodel.strat_col)
    
    structuralmodel.model = model
    
def evaluate_geomodel_scalarfield(structuralmodel, geomodel):
    # Evaluate scalar field
    SF = []
    for k, feat in enumerate(structuralmodel.sequences[1:]):
        print('k, feat = ', k, feat)
        values = structuralmodel.strat[structuralmodel.strat['sequence'] == feat].val.tolist()
        values = sorted(values)
        print(feat, values)
        V = structuralmodel.model.evaluate_feature_value(feat, np.array(geomodel.xyz))
        W = V.reshape((geomodel.nlay, geomodel.ncpl))  
        SF.append(W)

    return SF

    