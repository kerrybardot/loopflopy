import numpy as np
import flopy
import math
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.interpolate import griddata
from shapely.geometry import LineString
import geopandas as gpd
from scipy.interpolate import griddata
from matplotlib import gridspec
import loopflopy.utils as utils

logfunc = lambda e: np.log10(e)

def find_angle1_transect(nv, rotation_angle):
    """
    Calculate angle1 (dip direction) for transect models.
    Angle 1 rotates around z axis counterclockwise looking from +ve z (like a bearing).
    
    For 2D transect models, the dip direction is simply the rotation angle
    of the transect line.
    
    Parameters
    ----------
    nv : ndarray
        Normal vector to the geological surface (not used in transect mode).
    rotation_angle : float
        Rotation angle of the transect line in degrees.
    
    Returns
    -------
    float
        Dip direction angle in degrees.
    """
    return rotation_angle

def find_angle2_transect(nv, rotation_angle):
    """
    Calculate angle2 (dip angle) for transect models.
    Angle 2 rotates around y axis clockwise looking from +ve y (dip).

    For 2D transect models, calculates the dip angle by finding the angle
    between the surface normal vector and the transect plane.
    
    Parameters
    ----------
    nv : ndarray
        Normal vector to the geological surface from Loop structural modeling.
    rotation_angle : float
        Rotation angle of the transect line in degrees.
    
    Returns
    -------
    float
        Dip angle in degrees.
    
    Notes
    -----
    Uses cross product between surface normal and transect normal to
    calculate the dip angle in the transect plane.
    """

    # normal vector (nv) to tangent plane (from Loop)
    n = np.array([np.tan(math.radians(rotation_angle)), 1, 0])  # Normal vector to transect- check this!!
    cp = np.cross(nv, n)  # Cross product to find the plane normal
    angle2 = np.degrees(math.atan(cp[2]/cp[0]))
    return angle2

# angle 1 (DIP DIRECTION) rotates around z axis counterclockwise looking from +ve z.
def find_angle1(nv):
    """
    Calculate angle1 (dip direction) from surface normal vector.
    Angle 1 rotates around z axis counterclockwise looking from +ve z (like a bearing).
    
    Computes the dip direction angle, which is the azimuth of the steepest
    descent direction on a geological surface.
    
    Parameters
    ----------
    nv : ndarray
        Normal vector to the geological surface [a, b, c].
    
    Returns
    -------
    float
        Dip direction angle in degrees (0-360), measured clockwise from north.
    
    Notes
    -----
    The dip direction is calculated by:
    1. Finding the steepest descent gradient in the x-y plane
    2. Computing the azimuth angle using atan2
    
    This angle represents the direction a ball would roll down the surface.
    """
    a, b, c = nv[0], nv[1], nv[2]

    # Find steepest descent gradient (g) in the x-y plane (Angle 1)
    g = np.array([-a/c, b/c])
    angle1 = np.degrees(np.arctan2(g[1], g[0])) # angle in xy plane (anticlockwise)
    return angle1

def find_angle2(nv):
    """
    Calculate angle2 (dip angle) from surface normal vector.
    Angle 2 rotates around y axis clockwise looking from +ve y (dip).
    
    Computes the dip angle, which is the angle between the geological
    surface and the horizontal plane.
    
    Parameters
    ----------
    nv : ndarray
        Normal vector to the geological surface [a, b, c].
    
    Returns
    -------
    float
        Dip angle in degrees (0-90), where 0 is horizontal and 90 is vertical.
    
    Notes
    -----
    The dip angle is calculated using the relationship between the
    horizontal and vertical components of the surface normal vector.
    This represents how steeply the surface is inclined.
    """
    a, b, c = nv[0], nv[1], nv[2]

    # Find steepest descent gradient (g) in the x-y plane (Angle 1)
    g = np.array([-a/c, b/c])

    # Find the Dip angle in the z direction (Angle 2)
    magnitude = np.linalg.norm(g)
    angle2 = np.degrees(np.arctan(np.sqrt(a**2 + b**2)/np.abs(c)))
    return angle2

def reshape_loop2mf(array, nlay, ncpl):
    array = array.reshape((nlay, ncpl))
    array = np.flip(array, 0)
    return(array)
    
class Geomodel:
    
    def __init__(self, scenario, mesh, structuralmodel, vertgrid, z0, z1, transect = False, nlg = None, **kwargs):     
           
        self.scenario = scenario                      
        self.mesh = mesh
        self.structuralmodel = structuralmodel
        self.vertgrid = vertgrid     
        self.z0 = z0
        self.z1 = z1
        self.transect = transect
        self.nlg = nlg # option to only use a subset of geological layers (from top)
        

        for key, value in kwargs.items():
            setattr(self, key, value)    
        
#---------- FUNCTION TO EVALUATE GEO MODEL AND POPULATE HYDRAULIC PARAMETERS ------#

    def evaluate_structuralmodel(self, mesh, structuralmodel): # Takes the project parameters and model class.         
        #print('   Creating Geomodel (lithology and discretisation arrays) for ', self.scenario, ' ...')
        self.units = np.array(structuralmodel.strat_names[1:])  
        self.strat_names = structuralmodel.strat_names[1:]
        if self.nlg is None:
            self.nlg = len(self.strat_names)  
        z0, z1 = self.z0, self.z1
        self.ncpl = mesh.ncpl


#---------- VOX - DIS ARRAY ------#

        if self.vertgrid == 'vox':
            
            self.nlay = nlay
            self.dz = (z1 - z0) / self.nlay
            self.ncell3d = mesh.ncpl * self.nlay
            self.idomain = np.ones((self.nlay, mesh.ncpl)) 
            self.top = z1 * np.ones((mesh.ncpl), dtype=float)
            
            self.zc = np.arange(z0 + self.dz / 2, z1, self.dz)  # Cell centres
            self.zbot = np.arange(z1 - self.dz, z0 - self.dz, -self.dz)
            
            self.botm = np.zeros((self.nlay, mesh.ncpl)) 
            for lay in range(self.nlay):
                self.botm[lay,:] = self.zbot[lay]

            #----- VOX - LITH AND VF ------#

            xyz = []                         
            for k in range(self.nlay):
                z = self.zc[k]
                for i in range(mesh.ncpl):    
                    x, y = mesh.xcyc[i][0], mesh.xcyc[i][1]
                    xyz.append([x,y,z])
            
            litho = structuralmodel.model.evaluate_model(xyz)  # generates an array indicating lithology for every cell
            vf = structuralmodel.model.evaluate_model_gradient(xyz) # generates an array indicating gradient for every cell
            
            # Reshape to lay, ncpl   
            litho = np.asarray(litho)
            litho = litho.reshape((self.nlay, mesh.ncpl))
            litho = np.flip(litho, 0)
            self.lith = litho
            self.lith_disv = litho
            
            ang1, ang2 = [], []
            for i in range(len(vf)):  
                ang1.append(find_angle1(vf[i]))
                ang2.append(find_angle2(vf[i]))
            self.ang1  = reshape_loop2mf(np.asarray(ang1), nlay, mesh.ncpl)
            self.ang2  = reshape_loop2mf(np.asarray(ang2), nlay, mesh.ncpl)
            
#---------- CON AND CON2  Finding geological layers bottoms ------#

        if self.vertgrid == 'con' or self.vertgrid == 'con2' : # CREATING DIS AND NPF ARRAYS
            
            print('   0. Creating xyz array...')
            t0 = datetime.now()
            nlay = int((z1 - z0)/self.res)
            dz = (z1 - z0)/nlay # actual resolution
            self.dz = dz
            zc = np.arange(z0 + dz / 2, z1, dz)  # Cell centres
            xyz = []  
            for k in range(nlay):
                z = zc[k]
                for i in range(mesh.ncpl):    
                    x, y = mesh.xcyc[i][0], mesh.xcyc[i][1]
                    xyz.append([x,y,z])

            t1 = datetime.now()
            run_time = t1 - t0
            print('Time taken Block 0 (creating xyz array) = ', run_time.total_seconds())

            print('\n   1. Evaluating structural model...')
            t0 = datetime.now()

            print('len(xyz) = ', len(xyz))      
            litho = structuralmodel.model.evaluate_model(np.array(xyz))  # generates an array indicating lithology for every cell
            litho = np.asarray(litho)
            litho = litho.reshape((nlay, mesh.ncpl)) # Reshape to lay, ncpl
            litho = np.flip(litho, 0)
            self.litho = litho

            t1 = datetime.now()
            run_time = t1 - t0
            print('Time taken Block 1 (Evaluate model) = ', run_time.total_seconds())

    def create_model_layers(self, mesh, structuralmodel, surface):

        if self.vertgrid == 'con' or self.vertgrid == 'con2' : # CREATING DIS AND NPF ARRAYS    

            # ------------------------------------------
            print('\n   2. Creating geo model layers...')
            t0 = datetime.now()

            # Arrays for geo arrays
            top_geo     = 9999 * np.ones((mesh.ncpl), dtype=float) # empty array to fill in ground surface
            botm_geo    = np.zeros((self.nlg, mesh.ncpl), dtype=float) # bottom elevation of each geological layer
            thick_geo   = np.zeros((self.nlg, mesh.ncpl), dtype=float) # geo layer thickness
            idomain_geo = np.ones((self.nlg, mesh.ncpl), dtype=float)      # idomain array for each lithology
           
            print('ncpl = ', mesh.ncpl)
            print('nlg number of geo layers = ', self.nlg)

            # IDOMAIN
            for icpl in range(mesh.ncpl):
                present = np.unique(self.litho[:,icpl])
                for p in present:
                    if p >= 0 and p < self.nlg: # don't include above ground, or deep geo layers not in flow model
                        idomain_geo[p, icpl] = 1
            
            
            def make_surfaces(structuralmodel, mesh):
         
                # This loop is for each LITHOLOGY
                print(structuralmodel.model.bounding_box.nsteps)

                surfaces = []
                for i in range(len(structuralmodel.vals)-1): # Don't create surface for the bottom lithology   
                    feature = structuralmodel.sequences[i]
                    vals = [structuralmodel.vals[i]]
                    print(f'     \nCreating surface for feature: {feature}, lithid {structuralmodel.lithids[i]}, value: {structuralmodel.vals[i]}')
                    surface = structuralmodel.model[feature].surfaces(vals)[0]  # Get the first surface for the feature
                    print('number of vertices = ', surface.vertices.shape[0])
                    x = surface.vertices[:,0]
                    y = surface.vertices[:,1]
                    z = surface.vertices[:,2]

                    # Filter out rows with NaN or inf values
                    valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z) & ~np.isinf(x) & ~np.isinf(y) & ~np.isinf(z)
                    filtered_points = np.array([x[valid_mask], y[valid_mask]]).T
                    filtered_z = z[valid_mask]
                    
                    print('len filtered z ', len(filtered_z), ' i.e. without NaN or inf')
                    surface = griddata(filtered_points, filtered_z, (mesh.xc, mesh.yc), method='nearest')#'linear', fill_value=-1)
                    #print(np.unique(surface, return_counts=True))

                    surfaces.append(surface)
                return surfaces

            surfaces = make_surfaces(structuralmodel, mesh) # round surface is surface 0!

            top_geo = surface # Make top of geomodel equal to DEM topography

            if self.nlg == len(structuralmodel.lithids)-1: # Situation when model bottom is z0
                for i in range(1, self.nlg): # don't include groud level or bottom surface
                    botm_geo[i-1] = surfaces[i] # Bottom of geological layer
                    botm_geo[-1] = self.z0 # Bottom of the last geological layer is the model bottom
            else: # Situation when model bottom is bottom of geo layer
                for i in range(1, self.nlg+1): # don't include groud level
                    botm_geo[i-1] = surfaces[i] # Bottom of geological layer

            print('top_geo shape', top_geo.shape)
            print('botm_geo', botm_geo.shape)
            
            
            # --- Now we have to check that there are no cells with top < botm
            # --- If so, we need to replace thickness with nearest nieghbour
            def find_nearest_bestest_neighbour(mesh, thickness, cellid, top_lay, bot_lay):
                x, y = mesh.xc[cellid], mesh.yc[cellid]
                distance = np.array([np.sqrt((mesh.xc[i]-x)**2 + (mesh.yc[i]-y)**2) for i in range(mesh.ncpl)])
                
                # FIND THE NEAREST. Mask where distance == 0 (the cell itself)
                masked_distance = np.ma.masked_where(distance == 0, distance)

                # FIND THE BESTEST. Mask where groundsurface below bottom surface
                masked = np.ma.masked_where(thickness <=0 , masked_distance)
                
                # Find the minimum distance excluding the masked values
                idx = np.where(distance == np.min(masked))[0][0]

                return idx
                       
            # This method assumes no pinchouts
            count = 0
            for geolay in range(self.nlg):
                if geolay == 0:
                    for icpl in range(mesh.ncpl):
                        thickness = top_geo - botm_geo[geolay]
                        print('thickness top layer.shape ', thickness.shape)
                        if thickness[geolay, icpl] <= 0: # if the bottom if above ground surface...
                            nncell = find_nearest_bestest_neighbour(mesh, thickness, icpl, top_geo, botm_geo[geolay]) # find nearest neighbour..
                            botm_geo[geolay][icpl] = top_geo[icpl] - thickness[nncell] # make the negative thickness cell half the thickness as neighbouring cell                          
                            count += 1
                else:
                    print('geolay ', geolay)
                    thickness = botm_geo[geolay-1] - botm_geo[geolay]
                    print('thickness bot layer.shape ', thickness.shape)
                    for icpl in range(mesh.ncpl):
                        if thickness[icpl] <= 0: # if the bottom if above ground surface...
                            nncell = find_nearest_bestest_neighbour(mesh, thickness, icpl, botm_geo[geolay-1], botm_geo[geolay]) # find nearest neighbour..
                            botm_geo[geolay][icpl] = botm_geo[geolay-1][icpl] - thickness[nncell] # make the negative thickness cell have half the thickness as neighbouring cell
                            #print(f'Cell id {icpl} has negative thickness. Replacing geo bottom with cell {nncell}')
                            count += 1

            print('\nNumber of cells with negative thickness that have been converted= ', count)
            print('min thickness = ', np.min(top_geo - botm_geo[geolay]), 'max thickness = ', np.max(top_geo - botm_geo[geolay]),'\n')

            # NEED TO WORK ON THIS IN FUTURE, USER CAN CHOOSE WHAT TO DO FOR PINCHOUTS!!!
            # Or we can pinchout the layers where missing???
            #new_array = np.where(top_geo - botm_geo[0] <= 0, 0, 1)
            #idomain_geo[0] = new_array
            #for i in range(1,self.nlg):
            #    new_array = np.where(botm_geo[i-1] - botm_geo[i] <= 0, 0, 1)
            #    idomain_geo[i] = new_array # if the top of the layer is above the bottom of the layer, make idomain 0

            t1 = datetime.now()
            run_time = t1 - t0
            print('Time taken Block 2 create geomodel layers ', run_time.total_seconds())

                           
            # -------------------------------
            print('\n   3. Evaluating geo layer thicknesses...')
            t0 = datetime.now()

            for lay_geo in range(self.nlg):
                    if lay_geo == 0:
                        thick_geo[lay_geo, :] = top_geo - botm_geo[lay_geo,:]
                    else:
                        thick_geo[lay_geo, :] = botm_geo[lay_geo-1,:] - botm_geo[lay_geo,:]
                        
            self.top_geo = top_geo
            self.botm_geo = botm_geo  
            self.thick_geo = thick_geo    
            print('min thick ', np.min(self.thick_geo), 'max thick ', np.max(self.thick_geo))
            self.idomain_geo = idomain_geo 
            t1 = datetime.now()
            run_time = t1 - t0
            print('Time taken Block 3 tiny bit', run_time.total_seconds())
                    
            
#----- CON - CREATE LITH, BOTM AND IDOMAIN ARRAYS (PILLAR METHOD, PICKS UP PINCHED OUT LAYERS) ------#    
        if self.vertgrid == 'con':

            print('\n   4. Creating flow model layers...')
            t0 = datetime.now()

            self.nlay   = self.nlg * self.nls # number of model layers = geo layers * sublayers 
            botm        = np.zeros((self.nlay, mesh.ncpl), dtype=float) # bottom elevation of each model layer
            idomain     = np.ones((self.nlay, mesh.ncpl), dtype=int)    # idomain for each model layer

            for icpl in range(mesh.ncpl): 
                for lay_geo in range(self.nlg):
                    for lay_sub in range(self.nls):
                        lay = lay_geo * self.nls + lay_sub
                        if idomain_geo[lay_geo, icpl] == 0: # if pinched out geological layer...
                            #print('pinchout! ', lay_geo, icpl)
                            idomain[lay, icpl] = -1          # model cell idomain = -1

            # Creates bottom of model layers
            lay_geo = 0 # Start with top geological layer
            botm[0,:] = self.top_geo - (self.top_geo - botm_geo[0])/self.nls # Very first model layer
            
            for i in range(1, self.nls): # First geo layer. i represent sublay 0,1,2 top down within layer
                lay = lay_geo * self.nls + i
                botm[lay,:] = self.top_geo - (i+1) * (self.top_geo - botm_geo[0])/self.nls
            for lay_geo in range(1, self.nlg): # Work through subsequent geological layers
                for i in range(self.nls): 
                    lay = lay_geo * self.nls + i
                    botm[lay,:] = botm_geo[lay_geo-1] - (i+1) * (botm_geo[lay_geo-1] - botm_geo[lay_geo])/self.nls
            
            self.lith  = np.ones_like(botm, dtype = float)
            for lay_geo in range(self.nlg):
                for i in range(self.nls):
                    lay = lay_geo * self.nls + i 
                    self.lith[lay,:] *= lay_geo
                    
            #self.botm_geo = botm_geo
            self.top = self.top_geo
            self.botm = botm
            self.idomain = idomain
            self.nlay = self.nlg * self.nls
            self.lith_disv = self.lith
            self.model_layers = [] # This creates a list of flow model layers for every geological
            for i in range(self.nlg):
                a = []
                for j in range(self.nls):
                    a.append(i * self.nls + j)
                self.model_layers.append(a)

            t1 = datetime.now()
            run_time = t1 - t0
            print('Time taken Block 4 create flow model layers = ', run_time.total_seconds())
 
        #----- CON - CREATE LITH, BOTM AND IDOMAIN ARRAYS (PILLAR METHOD, PICKS UP PINCHED OUT LAYERS) ------#    
        if self.vertgrid == 'con2':

            sublays     = np.zeros((self.nlg, mesh.ncpl), dtype=float) # number of sublayers
            dz_sublays  = np.zeros((self.nlg, mesh.ncpl), dtype=float) # geo layer thickness
            
            for lay_geo in range(self.nlg):
                for icpl in range(mesh.ncpl):
                    max_lay_thick = self.max_thick[lay_geo]
                    if thick_geo[lay_geo, icpl]/2 > max_lay_thick:
                        sublays[lay_geo, icpl] = math.ceil(thick_geo[lay_geo, icpl]/ max_lay_thick) # geo layer has a minimum of 2 model layers per geo layer
                    else: 
                        sublays[lay_geo, icpl]= 2 # geo layer has a minimum of 2 model layers per geo layer
                    dz_sublays[lay_geo, icpl] = thick_geo[lay_geo, icpl] / sublays[lay_geo, icpl]
                        
            max_sublays = np.ones((self.nlg),  dtype=int)
            for lay_geo in range(self.nlg):
                max_sublays[lay_geo] = sublays[lay_geo, :].max()
            nlay = max_sublays.sum()     
            
            # Arrays for flow model
            botm        = np.zeros((nlay, mesh.ncpl), dtype=float) # bottom elevation of each model layer
            lith        = np.zeros((nlay, mesh.ncpl), dtype=float) # bottom elevation of each model layer
            idomain     = np.ones((nlay, mesh.ncpl), dtype=int)    # idomain for each model layer
            
            # Here we make bottom arrays - pinched out cells have the same bottom as layer above
            for icpl in range(mesh.ncpl):
                lay = 0 # model layer
                for lay_geo in range(self.nlg):
                    #if icpl == 500: print('GEO LAY = ', lay_geo)
                    nsublay    = sublays[lay_geo, icpl]
                    dz         = thick_geo[lay_geo, icpl] / nsublay
                    max_sublay = max_sublays[lay_geo]
                    for s in range(max_sublay): # marches through each sublayer of geo layer
                        if s < nsublay: # active cell
                            if lay == 0:
                                #if icpl == 500: print('Top layer, lay = ', lay)
                                #if icpl == 500: print(top[icpl] - dz)
                                botm[lay, icpl] = self.top_geo[icpl] - dz
                                lith[lay, icpl] = lay_geo
                            else:
                                #if icpl == 500: print('Not top layer, lay = ', lay)
                                #if icpl == 500: print(botm[lay-1, icpl] - dz)
                                botm[lay, icpl] = botm[lay-1, icpl] - dz
                                lith[lay, icpl] = lay_geo
 
                        else:  # pinched out cell
                            #if icpl == 500: print('PINCHOUT, lay = ', lay)
                            botm[lay, icpl] = botm[lay-1, icpl] # use the bottom before it
                            lith[lay, icpl] = lay_geo
                        lay += 1 # increase mode layer by 1
                        
            # Now we make idomain so that pinched out cells have an idomain of -1
            for icpl in range(mesh.ncpl):
                if botm[0, icpl] == self.top_geo[icpl]:
                    #print(icpl)
                    #print(icpl, botm[0, icpl], self.top_geo[icpl])
                    idomain[0, icpl] = -1
                for lay in range(1,nlay):
                    if botm[lay, icpl] == botm[lay-1, icpl]:
                        idomain[lay, icpl] = -1
    
            self.top = self.top_geo
            self.botm_geo = botm_geo      
            self.botm = botm
            self.idomain = idomain
            self.lith = lith
            self.lith_disv = lith
            self.nlay = nlay



        ###############
                #---------CON/CON2 Calculating cellids and gradients
        if self.vertgrid == 'con' or self.vertgrid == 'con2':

            # Sort out cell ids
            # First create an array for cellids in layered version  (before we pop cells that are absent)
            self.cellid_disv = np.empty_like(self.lith_disv, dtype = int)
            self.cellid_disu = -1 * np.ones_like(self.lith_disv, dtype = int)
            i = 0
            for lay in range(self.nlay):
                for icpl in range(mesh.ncpl):
                    self.cellid_disv[lay, icpl] = lay * mesh.ncpl + icpl
                    if self.idomain[lay, icpl] != -1:
                        self.cellid_disu[lay, icpl] = i
                        i += 1
            self.ncell_disv = self.cellid_disv.size
            self.ncell_disu = np.count_nonzero(self.cellid_disu != -1)

            # Calculate gradients
            xyz = []                         
            for lay in range(self.nlay-1, -1, -1):
                for icpl in range(mesh.ncpl):  
                    x, y, z = mesh.xcyc[icpl][0], mesh.xcyc[icpl][1], self.botm[lay, icpl] 
                    xyz.append([x,y,z])
            vf = structuralmodel.model.evaluate_model_gradient(xyz) # generates an array indicating gradient for every cell
            
            ang1, ang2 = [], []
            if self.transect:
                for i in range(len(vf)):  
                    ang1.append(find_angle1_transect(vf[i], mesh.angrot))
                    ang2.append(find_angle2_transect(vf[i], mesh.angrot))
            else:
                for i in range(len(vf)):  
                    ang1.append(find_angle1(vf[i]))
                    ang2.append(find_angle2(vf[i]))

            self.ang1  = reshape_loop2mf(np.asarray(ang1), self.botm.shape[0], self.botm.shape[1])
            self.ang2  = reshape_loop2mf(np.asarray(ang2), self.botm.shape[0], self.botm.shape[1])



        self.nnodes_div = len(self.botm.flatten())   
        self.thick = np.zeros((self.nlay, mesh.ncpl))
        self.thick[0,:] = self.top_geo - self.botm[0]
        self.thick[1:-1,:] = self.botm[0:-2] - self.botm[1:-1]
        self.thick.min()
        self.zc = self.botm + self.thick/2

        self.vgrid = flopy.discretization.VertexGrid(vertices=mesh.vertices, cell2d=mesh.cell2d, ncpl = mesh.ncpl, 
                                                     top = self.top_geo[0], botm = self.botm)
    
        # Save xyz coordinates for each cell in the model       
        xyz = []  
        for lay in range(self.nlay):
            z_lay = self.zc[lay] # z coordinate for the layer
            for icpl in range(mesh.ncpl):    
                x, y = mesh.xcyc[icpl][0], mesh.xcyc[icpl][1]
                z = z_lay[icpl]
                xyz.append([x,y,z])
        self.xyz = xyz

        t1 = datetime.now()
        run_time = t1 - t0
        print('Time taken Block 5 gradients= ', run_time.total_seconds())



################## PROP ARRAYS TO BE SAVED IN DISU FORMAT ##################        
    def fill_cell_properties(self, mesh): # Uses lithology codes to populate arrays 

        
        
        '''# First create an array for cellids in layered version  (before we pop cells that are absent)
        self.cellid_disv = np.empty_like(self.lith_disv, dtype = int)
        self.cellid_disu = -1 * np.ones_like(self.lith_disv, dtype = int)
        i = 0
        for lay in range(self.nlay):
            for icpl in range(mesh.ncpl):
                self.cellid_disv[lay, icpl] = lay * mesh.ncpl + icpl
                if self.idomain[lay, icpl] != -1:
                    self.cellid_disu[lay, icpl] = i
                    i += 1
        self.ncell_disv = self.cellid_disv.size
        self.ncell_disu = np.count_nonzero(self.cellid_disu != -1)'''
        
#---------- PROP ARRAYS (VOX and CON) ----- 
# 
        print('   6. Filling cell properties...')
        t0 = datetime.now()  
        self.k11    = np.empty_like(self.lith_disv, dtype = float)
        self.k22    = np.empty_like(self.lith_disv, dtype = float)
        self.k33    = np.empty_like(self.lith_disv, dtype = float)
        self.ss     = np.empty_like(self.lith_disv, dtype = float)
        self.sy     = np.empty_like(self.lith_disv, dtype = float)
        self.iconvert = np.empty_like(self.lith_disv, dtype = float)

        for n in range(self.nlg):  # replace lithologies with parameters
            self.k11[self.lith_disv==n] = self.hk_perlay[n] 
            self.k22[self.lith_disv==n] = self.hk_perlay[n] 
            self.k33[self.lith_disv==n] = self.vk_perlay[n] 
            self.ss[self.lith_disv==n]  = self.ss_perlay[n]
            self.sy[self.lith_disv==n]  = self.sy_perlay[n]
            self.iconvert[self.lith_disv==n]  = self.iconvert_perlay[n]
                   
        # Force all K tensor angles in fault zone to 0 (Loop can't calculate angles in faulted area properly yet!)
        '''if 'spatial.fault_poly' in globals(): #if hassattr(P,"fault_poly"):
            for icpl in range(mesh.ncpl):
                point = Point(mesh.xcyc[icpl])
                if spatial.fault_poly.contains(point):
                    for lay in range(self.nlay):
                        self.ang1[lay,icpl] = 0  
                        self.ang2[lay,icpl] = 0 '''  
        ######################################
        
        self.lith   = self.lith_disv[self.cellid_disu != -1].flatten()
        self.k11    = self.k11[self.cellid_disu != -1].flatten()
        self.k22    = self.k22[self.cellid_disu != -1].flatten()
        self.k33    = self.k33[self.cellid_disu != -1].flatten()
        self.ss     = self.ss[self.cellid_disu != -1].flatten()
        self.sy     = self.sy[self.cellid_disu != -1].flatten()
        self.iconvert     = self.iconvert[self.cellid_disu != -1].flatten()
        print('ang1 shape ', self.ang1.shape)
        print(self.cellid_disu[self.cellid_disu != -1].size)
        #a1, a2 = self.ang1.reshape((self.nlay, self.ncpl)), self.ang2.reshape((self.nlay, self.ncpl))
        self.angle1 = self.ang1[self.cellid_disu != -1].flatten()
        self.angle2 = self.ang2[self.cellid_disu != -1].flatten()
        self.angle3 = np.zeros_like(self.angle1, dtype = float)  # Angle 3 always at 0
        
        print('angle1 shape ', self.angle1.shape)
        self.logk11    = logfunc(self.k11)
        self.logk22    = logfunc(self.k22)
        self.logk33    = logfunc(self.k33)
        
        t1 = datetime.now()
        run_time = t1 - t0
        print('Time taken Block 6 Fill cell properties = ', run_time.total_seconds())

    def fill_cell_properties_heterogeneous(self, properties): # Uses lithology codes to populate arrays 
       
#---------- PROP ARRAYS (VOX and CON) ----- 
# 
        print('   6. Filling cell properties...')
        t0 = datetime.now()  

        # Lith
        self.lith   = self.lith_disv[self.cellid_disu != -1].flatten()

        # iconvert
        self.iconvert = np.empty_like(self.lith_disv, dtype = float)
        for n in range(self.nlg): 
            self.iconvert[self.lith_disv==n]  = self.iconvert_perlay[n]
        self.iconvert     = self.iconvert[self.cellid_disu != -1].flatten()
                   
        # Force all K tensor angles in fault zone to 0 (Loop can't calculate angles in faulted area properly yet!)
        '''if 'spatial.fault_poly' in globals(): #if hassattr(P,"fault_poly"):
            for icpl in range(mesh.ncpl):
                point = Point(mesh.xcyc[icpl])
                if spatial.fault_poly.contains(point):
                    for lay in range(self.nlay):
                        self.ang1[lay,icpl] = 0  
                        self.ang2[lay,icpl] = 0 '''  
        ######################################
        
        # Heterogeneous properties
        self.k11    = properties.kh_disu
        self.k22    = properties.kh_disu
        self.k33    = properties.kv_disu
        self.ss     = properties.ss_disu
        self.sy     = properties.sy_disu
        
        print('ang1 shape ', self.ang1.shape)
        print(self.cellid_disu[self.cellid_disu != -1].size)
        #a1, a2 = self.ang1.reshape((self.nlay, self.ncpl)), self.ang2.reshape((self.nlay, self.ncpl))
        self.angle1 = self.ang1[self.cellid_disu != -1].flatten()
        self.angle2 = self.ang2[self.cellid_disu != -1].flatten()
        self.angle3 = np.zeros_like(self.angle1, dtype = float)  # Angle 3 always at 0
        
        print('angle1 shape ', self.angle1.shape)
        self.logk11    = logfunc(self.k11)
        self.logk22    = logfunc(self.k22)
        self.logk33    = logfunc(self.k33)
        
        t1 = datetime.now()
        run_time = t1 - t0
        print('Time taken Block 6 Fill cell properties = ', run_time.total_seconds())

    def geomodel_plan_lith(self, spatial, mesh, structuralmodel, **kwargs):
        x0 = kwargs.get('x0', spatial.x0)
        y0 = kwargs.get('y0', spatial.y0)
        x1 = kwargs.get('x1', spatial.x1)
        y1 = kwargs.get('y1', spatial.y1)

        fig = plt.figure(figsize = (10, 6))
        ax = plt.subplot(111)

        ax.set_aspect('equal')
        ax.set_title('surface geology', size = 10)

        mapview = flopy.plot.PlotMapView(modelgrid=self.vgrid, layer = 0, ax = ax)
        
        plan = mapview.plot_array(self.surf_lith, cmap=structuralmodel.cmap, norm = structuralmodel.norm, 
                                  alpha=0.8, ax = ax)
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('y (m)', size = 10)
        
        labels = structuralmodel.strat_names[1:]
        ticks = [i for i in np.arange(0,len(labels))]
        boundaries = np.arange(-1,len(labels),1)+0.5       
        
                # Create interpolation grid
        x = np.linspace(spatial.x0-100, spatial.x1+100, 1000)
        y = np.linspace(spatial.y0-100, spatial.y1+100, 1000)
        X, Y = np.meshgrid(x, y)
        Z = griddata((mesh.xc, mesh.yc), self.surf_lith, (X, Y), method='linear')

        # Contours
        levels = np.arange(0, self.nlg+1, 1)
        contour = ax.contour(X, Y, Z, levels = levels, extend = 'both', colors = 'Black', 
                   linewidths=1., linestyles = 'solid')
        
        cbar = plt.colorbar(plan,
                    boundaries = boundaries,
                    shrink = 1.0)
        cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')   
        plt.tight_layout()  
        plt.show()

        # Extract contour lines
        contour_lines = []
        for collection in contour.collections:
            for path in collection.get_paths():
                v = path.vertices
                contour_lines.append(LineString(v))

        #save_contour_lines_to_shapefile
        gdf = gpd.GeoDataFrame(geometry=contour_lines, crs = spatial.epsg)
        gdf.to_file('../data/data_shp/geomodel_surface_contours.shp', driver='ESRI Shapefile')

    def geomodel_transect_lith(self, plot_node = None, **kwargs):
 
        x0 = kwargs.get('x0', self.mesh.vgrid.xcellcenters[0])
        y0 = kwargs.get('y0', self.mesh.vgrid.ycellcenters[-1])
        x1 = kwargs.get('x1', self.mesh.vgrid.xcellcenters[0])
        y1 = kwargs.get('y1', self.mesh.vgrid.ycellcenters[-1])
        z0 = kwargs.get('z0', self.z0)
        z1 = kwargs.get('z1', self.z1)
    
        fig = plt.figure(figsize = (12,4))
        ax = plt.subplot(111)
        #ax.set_aspect('equal')
        xsect = flopy.plot.PlotCrossSection(modelgrid=self.vgrid , line={"line": [(x0, y0),(x1, y1)]}, geographic_coords=True)
        csa = xsect.plot_array(a = self.lith, cmap = self.structuralmodel.cmap, norm = self.structuralmodel.norm, alpha=0.8)
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('z (m)', size = 10)
        ax.set_ylim([z0, z1])
  
        linecollection = xsect.plot_grid(lw = 0.1, color = 'black') 

        if plot_node != None:
            x, y, z = utils.disucell_to_xyz(self, plot_node)
            ax.plot(x, z)
        
        labels = self.structuralmodel.strat_names[1:]
        ticks = [i for i in np.arange(0,len(labels))]
        boundaries = np.arange(-1,len(labels),1)+0.5

        cbar = plt.colorbar(csa,
                            boundaries = boundaries,
                            shrink = 1.0
                            )
        cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')    
        plt.title(f"x0, y0 = {x0:.0f}, {x1:.0f}: x1, y1 = {y0:.0f}, {y1:.0f}", size=8)
        plt.tight_layout()  
        plt.show()    

    def get_surface_lith(self):
        lith = self.lith_disv
        idomain = self.idomain
        ncpl = lith.shape[1]
        surf_lith = -999 * np.ones((self.lith_disv.shape[1]))
        for icpl in range(ncpl):
            for lay in range(self.lith_disv.shape[0]): # number of model layers
                if idomain[lay, icpl] == 1:                # if present
                    surf_lith[icpl] = lith[lay, icpl] 
                    break
        self.surf_lith = surf_lith

    def contour_bottom(self, spatial, mesh, structuralmodel, unit, contour_interval):
        """
        Contour the bottom of a unit
        """
        def rounded_down(number, contour_interval): # rounded_to is the nearest 1, 10, 100 etc
            return math.floor(number / contour_interval) * contour_interval
        def rounded_up(number, contour_interval): # rounded_to is the nearest 1, 10, 100 etc
            return math.ceil(number / contour_interval) * contour_interval
        
        #geo_lay = self.strat_names[self.strat_names == unit]
        geo_lay = np.where(np.array(self.strat_names) == unit)[0][0]

        print(unit, geo_lay)

        # Create interpolation grid
        x = np.linspace(spatial.x0-100, spatial.x1+100, 1000)
        y = np.linspace(spatial.y0-100, spatial.y1+100, 1000)
        X, Y = np.meshgrid(x, y)
        Z = griddata((mesh.xc, mesh.yc), self.botm_geo[geo_lay], (X, Y), method='linear')
        
        fig, ax = plt.subplots(figsize = (7,7))
        ax.set_title(f'Bottom elevation of {unit} from geo model')
        
        x, y = spatial.model_boundary_poly.exterior.xy
        ax.plot(x, y, '-o', ms = 2, lw = 1, color='black')
        x, y = spatial.inner_boundary_poly.exterior.xy
        ax.plot(x, y, '-o', ms = 2, lw = 0.5, color='black')

        # Plot raw data points
        df = structuralmodel.data[structuralmodel.data['lithcode'] == unit]
        ax.plot(df.X, df.Y, 'o', ms = 1, color = 'red')

        # Contours
        levels = np.arange(rounded_down(self.botm_geo[geo_lay].min(), contour_interval), 
                        rounded_up(self.botm_geo[geo_lay].max(),contour_interval)+contour_interval, 
                        contour_interval)
        #ax.contour(X, Y, Z, levels = levels, extend = 'both', colors = 'Black', linewidths=1., linestyles = 'solid')
        c = ax.contourf(X, Y, Z, levels = levels, extend = 'both', cmap='coolwarm', alpha = 0.5)
        ax.clabel(c, colors = 'black', inline=True, fontsize=8, fmt="%.0f")
        plt.colorbar(c, ax = ax, shrink = 0.5)
        
    def contour_surface(self, spatial, mesh, structuralmodel, contour_interval):
        """
        Contour the model top
        """
        def rounded_down(number, contour_interval): # rounded_to is the nearest 1, 10, 100 etc
            return math.floor(number / contour_interval) * contour_interval
        def rounded_up(number, contour_interval): # rounded_to is the nearest 1, 10, 100 etc
            return math.ceil(number / contour_interval) * contour_interval
        
        # Create interpolation grid
        x = np.linspace(spatial.x0-100, spatial.x1+100, 1000)
        y = np.linspace(spatial.y0-100, spatial.y1+100, 1000)
        X, Y = np.meshgrid(x, y)
        Z = griddata((mesh.xc, mesh.yc), self.top_geo, (X, Y), method='linear')
        
        fig, ax = plt.subplots(figsize = (7,7))
        ax.set_title(f'Top elevation of geo model')
        
        x, y = spatial.model_boundary_poly.exterior.xy
        ax.plot(x, y, '-o', ms = 2, lw = 1, color='black')
        x, y = spatial.inner_boundary_poly.exterior.xy
        ax.plot(x, y, '-o', ms = 2, lw = 0.5, color='black')
        
        # Plot raw data points
        df = structuralmodel.data[structuralmodel.data['lithcode'] == 'Ground']
        ax.plot(df.X, df.Y, 'o', ms = 1, color = 'red')

        # Contours
        levels = np.arange(rounded_down(self.top_geo.min(), contour_interval), 
                        rounded_up(self.botm_geo.max(),contour_interval)+contour_interval, 
                        contour_interval)
        #ax.contour(X, Y, Z, levels = levels, extend = 'both', colors = 'Black', linewidths=1., linestyles = 'solid')
        c = ax.contourf(X, Y, Z, levels = levels, extend = 'both', cmap='coolwarm', alpha = 0.5)
        ax.clabel(c, colors = 'black', inline=True, fontsize=8, fmt="%.0f")
        plt.colorbar(c, ax = ax, shrink = 0.5)

    def plot_surface(self, array, xlim = None, ylim = None): # e.g xlim = [700000, 707500]

        fig = plt.figure(figsize=(7,5))
        spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 0.05], wspace=0.2)

        ax = fig.add_subplot(spec[0], aspect="equal") #plt.subplot(1, 1, 1, aspect="auto")
        if xlim: ax.set_xlim(xlim) 
        if ylim: ax.set_ylim(ylim) 
            
        pmv = flopy.plot.PlotMapView(ax = ax, modelgrid=self.vgrid)
        p = pmv.plot_array(array, alpha = 0.6,)# cmap = cmap, norm = norm)
        cbar_ax = fig.add_subplot(spec[1])
        cbar = fig.colorbar(p, cax=cbar_ax, shrink = 0.1)  # Center tick labels

        #plt.savefig('../figures/surface.png')

    def geomodel_transect_array(self, array, title, grid = True, figsize = (12,4),
                                vmin = None, vmax = None, **kwargs):
        """
        Plot a cross-sectional view of any 3D array through the geological model.
        
        Creates a 2D cross-section plot displaying values from a 3D array
        (such as hydraulic head, hydraulic conductivity, etc.) along a specified
        transect line through the model.
        
        Parameters
        ----------
        spatial : Spatial
            Spatial data object containing model boundaries.
        array : ndarray
            3D array to plot with shape (nlay, ncpl) matching the model grid.
        title : str
            Title for the plot and filename when saving.
        grid : bool, optional
            Whether to overlay model grid lines (default: True).
        vmin, vmax : float, optional
            Minimum and maximum values for color scale (default: None, auto-scale).
        **kwargs
            x0, y0 : float
                Starting coordinates of transect line (default: spatial bounds).
            x1, y1 : float
                Ending coordinates of transect line (default: spatial bounds).
            z0, z1 : float
                Vertical extent for plotting (default: model domain).
        
        Notes
        -----
        This is a general-purpose transect plotting method that can display:
        - Hydraulic head distributions
        - Hydraulic conductivity fields
        - Flow velocities
        - Any other 3D model property
        
        The plot is automatically saved to '../figures/geomodel_transect_{title}.png'
        """

        x0 = kwargs.get('x0', min(self.mesh.xc))
        y0 = kwargs.get('y0', min(self.mesh.yc))
        z0 = kwargs.get('z0', self.z0)
        x1 = kwargs.get('x1', max(self.mesh.xc))
        y1 = kwargs.get('y1', max(self.mesh.yc))
        z1 = kwargs.get('z1', self.z1)
    
        fig = plt.figure(figsize = figsize)
        ax = plt.subplot(111)
        xsect = flopy.plot.PlotCrossSection(modelgrid=self.vgrid , line={"line": [(x0, y0),(x1, y1)]}, geographic_coords=True)
        csa = xsect.plot_array(a = array, alpha=0.8, vmin = vmin, vmax = vmax)
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('z (m)', size = 10)        
        ax.set_ylim([z0, z1])
  
        if grid:
            linecollection = xsect.plot_grid(lw = 0.1, color = 'black') 
        
        cbar = plt.colorbar(csa, shrink = 1.0)
        plt.title(f"{title}\nx0, y0 = {x0:.0f}, {x1:.0f}: x1, y1 = {y0:.0f}, {y1:.0f}", size=8)
        plt.tight_layout()  
        plt.savefig(f'../figures/geomodel_transect_{title}.png')
        plt.show() 