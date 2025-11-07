# test

import numpy as np
import flopy
import math
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.interpolate import griddata
from shapely.geometry import LineString
import geopandas as gpd
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
    """
    Reshape and flip array from Loop format to MODFLOW format.
    
    Converts a 1D array from Loop structural modeling into a 2D array
    suitable for MODFLOW, with proper layer ordering.
    
    Parameters
    ----------
    array : ndarray
        1D array from Loop evaluation with length (nlay * ncpl).
    nlay : int
        Number of model layers.
    ncpl : int
        Number of cells per layer.
    
    Returns
    -------
    ndarray
        2D array with shape (nlay, ncpl), flipped so layer 0 is at top.
    
    Notes
    -----
    Loop structural modeling evaluates from bottom to top, while MODFLOW
    expects layers from top to bottom. This function handles the conversion.
    """
    array = array.reshape((nlay, ncpl))
    array = np.flip(array, 0)
    return(array)
    
class Geomodel:
    """
    A geological model class for creating 3D lithological block models for MODFLOW 6 simulations.
    
    This class converts LoopStructural structural geological models into discretized 3D models suitable for
    numerical groundwater flow modeling. It handles different vertical discretization schemes
    and manages hydraulic property assignment based on lithological units.
    
    Parameters
    ----------
    scenario : str
        Name identifier for the geological model scenario.
    vertgrid : str
        Vertical discretization scheme. Options:
        - 'vox': Fixed voxel-based discretization
        - 'con': Conformable layers with same number of sublayers everywhere (e.g. 2 flow model layers per geological layer)
        - 'con2': Conformable layers with variable sublayer thickness based on maximum thickness
    z0 : float
        Bottom elevation of the model domain (m). Note: code not updated yet to handle bottom of model being bottom of layer.
    z1 : float
        Top elevation of the model domain (m). Note that ground surface elevation will replace this value later.
    transect : bool, optional
        Whether this is a 2D transect model (default: False).
    nlg : int, optional
        Number of geological layers to use (default: None, uses all available). 
        Option to only use a subset of geological layers (from top)
    nlay : int, optional
        Number of model layers for 'vox' discretization (default: None).
    **kwargs
        Additional keyword arguments to set as instance attributes.
    
    Attributes
    ----------
    scenario : str
        Model scenario name.
    vertgrid : str
        Vertical discretization scheme (vox, con, con2)
    z0, z1 : float
        Bottom and top elevations of model domain.
    units : ndarray
        Array of geological unit names.
    strat_names : list
        List of stratigraphic unit names.
    ncpl : int
        Number of cells per layer.
    nlay : int
        Number of model layers.
    lith : ndarray
        Lithology codes for each model cell.
    k11, k22, k33 : ndarray
        Hydraulic conductivity tensors for each cell.
    ss, sy : ndarray
        Specific storage and specific yield for each cell.
    iconvert : ndarray
        Convertible layer flags for each cell.
    botm : ndarray
        Bottom elevations of flow model layers.
    top : ndarray
        Top elevation of top flow model layer.
    idomain : ndarray
        Active/inactive cell flags. 1=active, -1=inactive (gets pinched out).
    
    Examples
    --------
    >>> # Create a geological model with conformable layers
    >>> geomodel = Geomodel('test_scenario', 'con', z0=-500, z1=100, nlay=25)
    >>> 
    >>> # Evaluate structural model at mesh points
    >>> geomodel.evaluate_structuralmodel(mesh, structural_model, res=5)
    >>> 
    >>> # Create model layers
    >>> surface = np.ones(mesh.ncpl) * 100  # Ground surface elevation
    >>> geomodel.create_model_layers(mesh, structural_model, surface, nls=2)
    """
    
    def __init__(self, scenario, mesh, structuralmodel, vertgrid, z0, z1, transect = False, 
                 nlg = None, nlay = None, **kwargs):   
           
        self.scenario = scenario   
        self.mesh = mesh
        self.structuralmodel = structuralmodel                   
        self.vertgrid = vertgrid     
        self.z0 = z0
        self.z1 = z1
        self.transect = transect
        self.nlg = nlg # option to only use a subset of geological layers (from top)
        self.nlay = nlay

        for key, value in kwargs.items():
            setattr(self, key, value)    
        
#---------- FUNCTION TO EVALUATE GEO MODEL AND POPULATE HYDRAULIC PARAMETERS ------#

    def evaluate_structuralmodel(self, mesh, structuralmodel, res = None):
        """
        Evaluate the 3D structural geological model at mesh cell centers.
        
        This method queries the structural model at all mesh cell centers to determine
        lithology codes and gradient information. For 'con' and 'con2' discretization,
        it creates a high-resolution evaluation grid that is later used to create
        conformable model layers.
        
        Parameters
        ----------
        mesh : Mesh
            The computational mesh object containing cell centers and geometry.
        structuralmodel : StructuralModel
            The structural geological model to evaluate.
        res : float, optional
            Vertical resolution for 'con' and 'con2' discretization schemes (m).
            Not used for 'vox' scheme.
        
        Notes
        -----
        - For 'vox' discretization: Creates regular voxel grid and evaluates model
        - For 'con'/'con2' discretization: Creates high-resolution evaluation points
          for later layer boundary interpolation
        - Stores lithology codes and gradient vectors for each evaluation point
        - Gradient vectors are used to calculate hydraulic conductivity tensor angles
        
        Sets Attributes
        ---------------
        units : ndarray
            Geological unit names from structural model.
        strat_names : list
            Stratigraphic unit names.
        ncpl : int
            Number of cells per layer from mesh.
        plangrid : str
            Plan view grid type from mesh ('car' (cartesian), 'tri' (triangular) or 'vor' (Voronoi)).
        lith : ndarray
            Lithology codes at each model cell.
        litho : ndarray
            High-resolution lithology evaluation (for 'con'/'con2').
        ang1, ang2 : ndarray
            Hydraulic conductivity tensor angles used in NPF package.
        """         
        print('Creating Geomodel for ', self.scenario, ' ...\n')

        print('0. Creating xyz array... ')
        t0 = datetime.now()
        
        self.units = np.array(structuralmodel.strat_names[1:])
        self.strat_names = structuralmodel.strat_names[1:]
        if self.nlg is None:
            self.nlg = len(self.strat_names)  
        z0, z1 = self.z0, self.z1
        self.ncpl = mesh.ncpl
        self.plangrid = mesh.plangrid

#---------- EVALUATE STRUCTUAL MODEL - VOX  ------#
        if self.vertgrid == 'vox':

            self.dz = (z1 - z0) / self.nlay
            self.zc = np.arange(z0 + self.dz / 2, z1, self.dz)  # Cell centres

            xyz = []                         
            for k in range(self.nlay):
                z = self.zc[k]
                for i in range(mesh.ncpl):    
                    x, y = mesh.xcyc[i][0], mesh.xcyc[i][1]
                    xyz.append([x,y,z])
            
            print('   len(xyz) = ', len(xyz))  
            litho = structuralmodel.model.evaluate_model(xyz)  # generates an array indicating lithology for every cell
            vf = structuralmodel.model.evaluate_model_gradient(xyz) # generates an array indicating gradient for every cell

            # Reshape to lay, ncpl   
            litho = np.asarray(litho)
            litho = litho.reshape((self.nlay, mesh.ncpl))
            litho = np.flip(litho, 0)
            litho = np.where(litho == -1, 0, litho) # This is to fill "sky cells" with surface lithology. Need to fix this later.
            self.lith = litho
            self.lith_disv = litho
            
            ang1, ang2 = [], []
            if self.transect:
                for i in range(len(vf)):  
                    ang1.append(find_angle1_transect(vf[i], mesh.angrot))
                    ang2.append(find_angle2_transect(vf[i], mesh.angrot))
            else:
                for i in range(len(vf)):  
                    ang1.append(find_angle1(vf[i]))
                    ang2.append(find_angle2(vf[i]))
            self.ang1  = reshape_loop2mf(np.asarray(ang1), self.nlay, mesh.ncpl)
            self.ang2  = reshape_loop2mf(np.asarray(ang2), self.nlay, mesh.ncpl)
            
#---------- EVALUATE STRUCTUAL MODEL - CON and CON2  ------#

        if self.vertgrid == 'con' or self.vertgrid == 'con2' : # CREATING DIS AND NPF ARRAYS
            self.res = res
            print('resolution for evaluating structural model = ', res)
            print('z0 z1 self.res =', z0, z1, self.res)
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
            print('   Time taken Block 0 (creating xyz array) = ', run_time.total_seconds())

            print('\n1. Evaluating structural model... -----')
            t0 = datetime.now()
            self.xyz = xyz  
            print('   len(xyz) = ', len(xyz))      
            litho = structuralmodel.model.evaluate_model(np.array(xyz))  # generates an array indicating lithology for every cell
            litho = np.asarray(litho)
            litho = litho.reshape((nlay, mesh.ncpl)) # Reshape to lay, ncpl
            litho = np.flip(litho, 0)
            self.litho = litho
            
            t1 = datetime.now()
            run_time = t1 - t0
            print('   Time taken Block 1 (Evaluate model) = ', run_time.total_seconds())

    def create_model_layers(self, surface, max_thick = None, nls = None):
        """
        Create 3D model layer geometry from evaluated structural model.
        
        This method converts the high-resolution structural model evaluation into
        discrete model layers suitable for numerical flow simulation. Different
        algorithms are used depending on the vertical discretization scheme.
        
        Parameters
        ----------
        mesh : Mesh
            The computational mesh object.
        structuralmodel : StructuralModel
            The structural model created by LoopStructural
        surface : ndarray
            Ground surface elevation at each mesh cell (m).
        max_thick : list of float, optional
            Maximum thickness for each geological layer when using 'con2' discretization (m).
            Only used with 'con2' scheme.
        nls : int, optional
            Number of sublayers (flowmodel layers per geological layer) for 'con' and 'con2' discretization.
            Only used with 'con' and 'con2' schemes.
        
        Notes
        -----
        Discretization Schemes:
        
        - 'vox': Regular voxel grid with uniform layer thickness
        - 'con': Conformable layers with constant sublayer thickness per geological unit
        - 'con2': Conformable layers with variable sublayer thickness based on max_thick
        
        The method handles:
        - Layer pinchouts (cells with zero thickness)
        - Inactive cell identification (idomain = -1)
        - Bottom elevation calculation for each model layer
        - Cell ID mapping between different grid formats
        
        Sets Attributes
        ---------------
        nlay : int
            Total number of model layers.
        botm : ndarray
            Bottom elevations of model layers (nlay, ncpl).
        top : ndarray
            Top elevations of model layers (ncpl,).
        idomain : ndarray
            Active cell flags (nlay, ncpl). 1=active, -1=inactive.
        lith : ndarray
            Lithology codes for each model cell.
        thick_geo : ndarray
            Thickness of each geological layer (nlg, ncpl).
        cellid_disv, cellid_disu : ndarray
            Cell ID mappings for DISV and DISU grid formats.
        ncell_disv, ncell_disu : int
            Total number of cells in DISV and DISU formats.
        model_layers : list
            Mapping of model layers to geological layers.
        """

        self.max_thick = max_thick
        self.nls = nls

        print('\n2. Creating geo model layers...')
        t0 = datetime.now()
  
#---------- CREATE MODEL LAYERS - VOX  ------#
        if self.vertgrid == 'vox':

            self.idomain = np.ones((self.nlay, self.mesh.ncpl)) 
            self.top = self.z1 * np.ones((self.mesh.ncpl), dtype=float)
            
            self.zbot = np.arange(self.z1 - self.dz, self.z0 - self.dz, -self.dz)
            self.botm = np.zeros((self.nlay, self.mesh.ncpl)) 
            for lay in range(self.nlay):
                self.botm[lay,:] = self.zbot[lay]

            # Sort out cell ids
            self.cellid_disv = 1 * np.ones_like(self.botm, dtype = int)
            self.cellid_disu = -1 * np.ones_like(self.botm, dtype = int)
            i = 0
            for lay in range(self.nlay):
                for icpl in range(self.mesh.ncpl):
                    self.cellid_disv[lay, icpl] = lay * self.mesh.ncpl + icpl
                    if self.idomain[lay, icpl] != -1:
                        self.cellid_disu[lay, icpl] = i
                        i += 1
            self.ncell_disv = self.cellid_disv.size
            self.ncell_disu = np.count_nonzero(self.cellid_disu != -1)

        
#---------- CREATE MODEL LAYERS - CON and CON2  ------#
        if self.vertgrid == 'con' or self.vertgrid == 'con2' : # CREATING DIS AND NPF ARRAYS    
     
            def start_stop_arr(initial_list): # Function to look down pillar and pick geo bottoms
                a = np.asarray(initial_list)
                mask = np.concatenate(([True], a[1:] != a[:-1], [True]))
                idx = np.flatnonzero(mask)
                l = np.diff(idx)
                start = np.repeat(idx[:-1], l)
                stop = np.repeat(idx[1:]-1, l)
                return(start, stop)
            
            nlay = int((self.z1 - self.z0)/self.res)
            dz = (self.z1 - self.z0)/nlay # actual resolution
            
            # Arrays for geo arrays
            top_geo     = 9999 * np.ones((self.mesh.ncpl), dtype=float) # empty array to fill in ground surface
            botm_geo    = np.zeros((self.nlg, self.mesh.ncpl), dtype=float) # bottom elevation of each geological layer
            thick_geo   = np.zeros((self.nlg, self.mesh.ncpl), dtype=float) # geo layer thickness
            idomain_geo = np.zeros((self.nlg, self.mesh.ncpl), dtype=float)      # idomain array for each lithology
            
            stop_array = np.zeros((self.nlg+1, self.mesh.ncpl), dtype=float)
            #print('stop_array shape', stop_array .shape )
            V = self.litho
            W = V[1:, :] - V[:-1, :] 
            #print(np.unique(W, return_counts=True))

            if np.any(W < 0):
                #print(f'***Geology repeats itself {np.sum(W < 0)} times at {np.where(W < 0)}*****')
                W[W < 0] = 0 # this is to handle instances where geology goes in reverse order (back to a younger sequence)

            print('   nlay = ', nlay)
            print('   ncpl = ', self.mesh.ncpl)
            print('   nlg number of geo layers = ', self.nlg)

            for icpl in range(self.mesh.ncpl):
                #print('ICPL = ', icpl)
                #if icpl == 73:
                    #print('Pillar lithologies ', V[:,icpl])
                    #print('V - V ', V[1:, icpl] - V[:-1, icpl])
                    #print('W ', W[:, icpl])
                # IDOMAIN
                present = np.unique(V[:,icpl])
                for p in present:
                    if p >= 0: # don't include above ground 
                        idomain_geo[p, icpl] = 1
                
                stop = np.array([nlay-1]) # Add the last layer to start with
                #print('line 239 stop ', stop)
                for i in range(1,self.nlg): 
                    
                    idx = np.where(W[:,icpl] == i)[0] # checking if different from row above
                    # e.g. if i =  3 and idx =  [146 199], then it skips 3 layers TWICE!
                    if idx.size != 0: # If there returns an index
                        #print('geo layer = ', i, 'index where lith changes = ', idx)
                        
                        for id in idx:
                            #if icpl == 73: print('id = ', id, 'idx = ', idx, 'np.ones(i) = ', np.ones(i))
                            idx_array = id * np.ones(i)
                            stop = np.concatenate((stop, idx_array))
    
                n = self.nlg+1 - len(stop)# number of pinched out layers at the bottom not yet added to stop array
                #print('number of pinched out layers at the bottom not yet added to stop array = ', n)
                #print('stop  ', stop)
                m = (nlay-1) * np.ones(n) # this is just accounting for the bottom geo layers that dont exist in pillar
                stop = np.concatenate((stop, m))

                stop = np.sort(stop)
                #if icpl == 29: print('stop ', stop)     
                stop_array[:,icpl] = stop

            #print('stop_array.shape = ', stop_array.shape)
            botm_geo = self.z1 - (stop_array + 1)* self.dz
            botm_geo[:, 0]
            
            top_geo = surface
            botm_geo = botm_geo[1:,:]
            print('   botm_geo_shape', botm_geo.shape)

            for lay_geo in range(self.nlg):
                    if lay_geo == 0:
                        thick_geo[lay_geo, :] = top_geo - botm_geo[lay_geo,:]
                    else:
                        thick_geo[lay_geo, :] = botm_geo[lay_geo-1,:] - botm_geo[lay_geo,:]
                     
            self.top_geo = top_geo
            self.botm_geo = botm_geo  
            self.thick_geo = thick_geo    
            self.idomain_geo = idomain_geo 
        t1 = datetime.now()
        run_time = t1 - t0
        print('   Time taken Block 2 (Create geological model layers)', run_time.total_seconds())

        print('\n3. Creating flow model layers...')
        t0 = datetime.now()
        #----- CON - CREATE LITH, BOTM AND IDOMAIN ARRAYS (PILLAR METHOD, PICKS UP PINCHED_OUT LAYERS) ------#    
        if self.vertgrid == 'con':

            t0 = datetime.now()

            self.nlay   = self.nlg * self.nls # number of model layers = geo layers * sublayers 
            self.lith = np.zeros((self.nlay, self.mesh.ncpl), dtype=float) # lithology for each model layer
            self.botm = np.zeros((self.nlay, self.mesh.ncpl), dtype=float) # bottom elevation for each model layer
            self.idomain = np.zeros((self.nlay, self.mesh.ncpl), dtype=int) # idomain for each model layer

            botm        = np.zeros((self.nlay, self.mesh.ncpl), dtype=float) # bottom elevation of each model layer
            idomain     = np.ones((self.nlay, self.mesh.ncpl), dtype=int)    # idomain for each model layer

            for icpl in range(self.mesh.ncpl): 
                for lay_geo in range(self.nlg):
                    for lay_sub in range(self.nls):
                        lay = lay_geo * self.nls + lay_sub
                        if idomain_geo[lay_geo, icpl] == 0: #  if pinched out geological layer...
                            #print('pinchout! ', lay_geo, icpl)
                            idomain[lay, icpl] = -1          # model cell idomain = -1
                        if self.thick_geo[lay_geo, icpl] <= 0: # if geo layer thickness = 0 (pinched out)
                            idomain[lay, icpl] = -1          # model cell idomain = -1
                        if lay_geo >=1: # bottom geological layers not modelled
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
            self.top = self.top_geo # top of model layers is the same as top of geo layers
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
 
        #----- CON2 - CREATE LITH, BOTM AND IDOMAIN ARRAYS (PILLAR METHOD, PICKS UP PINCHED OUT LAYERS) ------#    
        if self.vertgrid == 'con2':

            sublays     = np.zeros((self.nlg, self.mesh.ncpl), dtype=float) # number of sublayers
            dz_sublays  = np.zeros((self.nlg, self.mesh.ncpl), dtype=float) # geo layer thickness
            
            for lay_geo in range(self.nlg):
                for icpl in range(self.mesh.ncpl):
                    max_lay_thick = self.max_thick[lay_geo]
                    if self.thick_geo[lay_geo, icpl]/self.nls > max_lay_thick:
                        sublays[lay_geo, icpl] = math.ceil(self.thick_geo[lay_geo, icpl]/ max_lay_thick) # geo layer has a minimum of 2 model layers per geo layer
                    else: 
                        sublays[lay_geo, icpl]= self.nls # geo layer has a minimum of 2 model layers per geo layer
                    dz_sublays[lay_geo, icpl] = self.thick_geo[lay_geo, icpl] / sublays[lay_geo, icpl]
                        
            max_sublays = np.ones((self.nlg),  dtype=int)
            for lay_geo in range(self.nlg):
                max_sublays[lay_geo] = sublays[lay_geo, :].max()
            nlay = max_sublays.sum()     
            #print('Max sublays: ', max_sublays)
            
            # Arrays for flow model
            botm        = np.zeros((nlay, self.mesh.ncpl), dtype=float) # bottom elevation of each model layer
            lith        = np.zeros((nlay, self.mesh.ncpl), dtype=float) # bottom elevation of each model layer
            idomain     = np.ones((nlay, self.mesh.ncpl), dtype=int)    # idomain for each model layer
            
            # Here we make bottom arrays - pinched out cells have the same bottom as layer above
            for icpl in range(self.mesh.ncpl):
                lay = 0 # model layer
                for lay_geo in range(1):#self.nlg): KB

                    nsublay    = sublays[lay_geo, icpl] # number of sublayers in geo layer
                    dz         = self.thick_geo[lay_geo, icpl] / nsublay
                    # AI suggestion
                    #if nsublay == 0: # if pinched out layer, skip   
                    #    idomain[lay, icpl] = -1
                    #    botm[lay, icpl] = botm[lay-1, icpl] if lay > 0 else self.top_geo[icpl]
                    #    lith[lay, icpl] = lay_geo
                    #    lay += 1
                    #    continue

                    for s in range(max_sublays[lay_geo]): # marches through each sublayer of geo layer
                        #print('   icpl: ', icpl, 'lay_geo: ', lay_geo, 'sublay: ', s)
                        if s < nsublay: # active cell
                            if lay == 0:
                                #if icpl == 500: print('Top layer, lay = ', lay)
                                #if icpl == 500: print(top[icpl] - dz)
                                botm[lay, icpl] = self.top_geo[icpl] - dz # KB
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
            for icpl in range(self.mesh.ncpl):
                if botm[0, icpl] == self.top_geo[icpl]:
                    #print(icpl)
                    #print(icpl, botm[0, icpl], self.top_geo[icpl])
                    idomain[0, icpl] = -1
                for lay in range(1,nlay):
                    if botm[lay, icpl] == botm[lay-1, icpl]:
                        idomain[lay, icpl] = -1
    
            self.botm_geo = botm_geo      
            #botm = botm.reshape((geomodel.nlay, mesh.nrow, mesh.ncol))
            self.botm = botm
            self.idomain = idomain
            self.lith = lith
            self.lith_disv = lith
            self.nlay = nlay
            self.top = self.top_geo # top of model layers is the same as top of geo layers


        t1 = datetime.now()
        run_time = t1 - t0
        print('   Time taken Block 3 create flow model layers = ', run_time.total_seconds())

        print('\n4. Calculating cellids and gradients...')
        t0 = datetime.now()
        
        #---------CON/CON2 Calculating cellids and gradients
        if self.vertgrid == 'con' or self.vertgrid == 'con2':

            # Sort out cell ids
            # First create an array for cellids in layered version  (before we pop cells that are absent)
            self.cellid_disv = np.empty_like(self.lith_disv, dtype = int)
            self.cellid_disu = -1 * np.ones_like(self.lith_disv, dtype = int)
            i = 0
            for lay in range(self.nlay):
                for icpl in range(self.mesh.ncpl):
                    self.cellid_disv[lay, icpl] = lay * self.mesh.ncpl + icpl
                    if self.idomain[lay, icpl] != -1:
                        self.cellid_disu[lay, icpl] = i
                        i += 1
            self.ncell_disv = self.cellid_disv.size
            self.ncell_disu = np.count_nonzero(self.cellid_disu != -1)

            # Calculate gradients
            xyz = []                         
            for lay in range(self.nlay-1, -1, -1):
                for icpl in range(self.mesh.ncpl):  
                    x, y, z = self.mesh.xcyc[icpl][0], self.mesh.xcyc[icpl][1], self.botm[lay, icpl] 
                    xyz.append([x,y,z])
            vf = self.structuralmodel.model.evaluate_model_gradient(xyz) # generates an array indicating gradient for every cell
            
            ang1, ang2 = [], []
            if self.transect:
                for i in range(len(vf)):  
                    ang1.append(find_angle1_transect(vf[i], self.mesh.angrot))
                    ang2.append(find_angle2_transect(vf[i], self.mesh.angrot))
            else:
                for i in range(len(vf)):  
                    ang1.append(find_angle1(vf[i]))
                    ang2.append(find_angle2(vf[i]))

            self.ang1  = reshape_loop2mf(np.asarray(ang1), self.botm.shape[0], self.botm.shape[1])
            self.ang2  = reshape_loop2mf(np.asarray(ang2), self.botm.shape[0], self.botm.shape[1])

        # Save model layer thicknesses as (lay, icpl) array        
        self.thick = np.zeros((self.nlay, self.mesh.ncpl))
        self.thick[0,:] = self.top - self.botm[0]
        self.thick[1:-1,:] = self.botm[0:-2] - self.botm[1:-1]

        self.nnodes_div = len(self.botm.flatten())   
        self.zc = self.botm + self.thick/2 # Save cell centres z value as (lay, icpl) array
        #self.vgrid = flopy.discretization.VertexGrid(vertices=self.mesh.vertices, cell2d=self.mesh.cell2d, ncpl = self.mesh.ncpl, 
        #                                             top = self.top, botm = self.botm)

        self.vgrid = flopy.discretization.VertexGrid(vertices=self.mesh.vertices, cell2d=self.mesh.cell2d, ncpl = self.mesh.ncpl, 
                                                top = self.top_geo[0], botm = self.botm)

        # Save xyz coordinates for each cell in the model       
        xyz = []  
        for lay in range(self.nlay):
            z_lay = self.zc[lay] # z coordinate for the layer
            for icpl in range(self.mesh.ncpl):    
                x, y = self.mesh.xcyc[icpl][0], self.mesh.xcyc[icpl][1]
                z = z_lay[icpl]
                xyz.append([x,y,z])
        self.xyz = xyz
        
        t1 = datetime.now()
        run_time = t1 - t0
        print('   Time taken Block 4 gradients= ', run_time.total_seconds())

################## PROP ARRAYS TO BE SAVED IN DISU FORMAT ##################        
    def fill_cell_properties(self, mesh):
        """
        Populate hydraulic property arrays based on lithology codes.
        
        This method assigns hydraulic properties (hydraulic conductivity, storage,
        convertibility) to each model cell based on its lithology code. Properties
        are read from the per-layer property lists and mapped to the 3D model.
        ## CURRENTLY UPGRADING THIS TO HANDLE KRIGING WITH PILOT POINTS ###
        
        Parameters
        ----------
        mesh : Mesh
            The computational mesh object.
        
        Notes
        -----
        The method:
        1. Creates property arrays for all cells
        2. Assigns properties based on lithology codes using per-layer values
        3. Handles inactive cells (sky cells) by setting properties to zero
        4. Calculates hydraulic conductivity tensor angles from gradient vectors
        5. Converts arrays from DISV to DISU format for unstructured grids
        6. Computes logarithmic hydraulic conductivity values
        
        Required Attributes
        -------------------
        hk_perlay : list
            Horizontal hydraulic conductivity for each geological layer (m/d).
        vk_perlay : list
            Vertical hydraulic conductivity for each geological layer (m/d).
        ss_perlay : list
            Specific storage for each geological layer (1/m).
        sy_perlay : list
            Specific yield for each geological layer (dimensionless).
        iconvert_perlay : list
            Convertible layer flags for each geological layer.
        
        Sets Attributes
        ---------------
        k11, k22, k33 : ndarray
            Hydraulic conductivity tensor components (DISU format).
        ss, sy : ndarray
            Storage properties (DISU format).
        iconvert : ndarray
            Convertible layer flags (DISU format).
        angle1, angle2, angle3 : ndarray
            Hydraulic conductivity tensor angles (DISU format).
        logk11, logk22, logk33 : ndarray
            Logarithmic hydraulic conductivity values.
        """ 
        
        print('\n5. Filling cell properties...')
        t0 = datetime.now()  
    
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
        self.k11    = np.empty_like(self.lith_disv, dtype = float)
        self.k22    = np.empty_like(self.lith_disv, dtype = float)
        self.k33    = np.empty_like(self.lith_disv, dtype = float)
        self.ss     = np.empty_like(self.lith_disv, dtype = float)
        self.sy     = np.empty_like(self.lith_disv, dtype = float)
        self.iconvert = np.empty_like(self.lith_disv, dtype = float)

        # Cells in the sky
        self.k11[self.lith_disv==-1] = 0
        self.k22[self.lith_disv==-1] = 0
        self.k33[self.lith_disv==-1] = 0
        self.ss[self.lith_disv==-1]  = 0
        self.sy[self.lith_disv==-1]  = 0
        self.iconvert[self.lith_disv==-1] = 0

        for n in range(self.nlg):  # replace lithologies with parameters
            self.k11[self.lith_disv==n] = self.hk_perlay[n] 
            self.k22[self.lith_disv==n] = self.hk_perlay[n] 
            self.k33[self.lith_disv==n] = self.vk_perlay[n] 
            self.ss[self.lith_disv==n]  = self.ss_perlay[n]
            self.sy[self.lith_disv==n]  = self.sy_perlay[n]
            self.iconvert[self.lith_disv==n]  = self.iconvert_perlay[n]
                   
        ######################################
        
        self.lith   = self.lith_disv[self.cellid_disu != -1].flatten()
        self.k11    = self.k11[self.cellid_disu != -1].flatten()
        self.k22    = self.k22[self.cellid_disu != -1].flatten()
        self.k33    = self.k33[self.cellid_disu != -1].flatten()
        self.ss     = self.ss[self.cellid_disu != -1].flatten()
        self.sy     = self.sy[self.cellid_disu != -1].flatten()
        self.iconvert     = self.iconvert[self.cellid_disu != -1].flatten()
        print('   ang1 shape ', self.ang1.shape)

        #a1, a2 = self.ang1.reshape((self.nlay, self.ncpl)), self.ang2.reshape((self.nlay, self.ncpl))
        self.angle1 = self.ang1[self.cellid_disu != -1].flatten()
        self.angle2 = self.ang2[self.cellid_disu != -1].flatten()
        self.angle3 = np.zeros_like(self.angle1, dtype = float)  # Angle 3 always at 0
        
        print('   angle1 shape ', self.angle1.shape)
        self.logk11    = logfunc(self.k11)
        self.logk22    = logfunc(self.k22)
        self.logk33    = logfunc(self.k33)
        
        t1 = datetime.now()
        run_time = t1 - t0
        print('   Time taken Block 5 Fill cell properties = ', run_time.total_seconds())

    def fill_cell_properties_heterogeneous(self, properties): # Uses lithology codes to populate arrays 
       
#---------- PROP ARRAYS (VOX and CON) ----- 
        print('\n5. Filling cell properties (heterogeneous)...')        
        t0 = datetime.now() 
        self.lith   = self.lith_disv[self.cellid_disu != -1].flatten() # Lith
        self.iconvert = np.empty_like(self.lith_disv, dtype = float) # iconvert
        for n in range(self.nlg): 
            self.iconvert[self.lith_disv==n]  = self.iconvert_perlay[n]
        self.iconvert     = self.iconvert[self.cellid_disu != -1].flatten()
                   
        ######################################
        
        # Heterogeneous properties
        self.k11    = properties.kh_disu
        self.k22    = properties.kh_disu
        self.k33    = properties.kv_disu
        self.ss     = properties.ss_disu
        self.sy     = properties.sy_disu
        
        print('   ang1 shape ', self.ang1.shape)
        print(self.cellid_disu[self.cellid_disu != -1].size)
        #a1, a2 = self.ang1.reshape((self.nlay, self.ncpl)), self.ang2.reshape((self.nlay, self.ncpl))
        self.angle1 = self.ang1[self.cellid_disu != -1].flatten()
        self.angle2 = self.ang2[self.cellid_disu != -1].flatten()
        self.angle3 = np.zeros_like(self.angle1, dtype = float)  # Angle 3 always at 0
        
        print('   angle1 shape ', self.angle1.shape)
        self.logk11    = logfunc(self.k11)
        self.logk22    = logfunc(self.k22)
        self.logk33    = logfunc(self.k33)
        
        t1 = datetime.now()
        run_time = t1 - t0
        print('   Time taken Block 5 Fill cell properties = ', run_time.total_seconds())

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

    def geomodel_transect_lith(self, figsize = (12,3), plot_node = None, **kwargs):
        """
        Plot a cross-sectional view of the geological model showing lithology.
        
        Creates a 2D cross-section plot displaying the lithological units along
        a specified transect line through the 3D geological model.
        
        Parameters
        ----------
        structuralmodel : StructuralModel
            The structural model containing color mapping and unit names.
        spatial : Spatial
            Spatial data object containing model boundaries.
        plot_node : int, optional
            Specific model node (zero-based) to highlight on the transect (default: None).
        **kwargs
            x0, y0 : float
                Starting coordinates of transect line (default: spatial bounds).
            x1, y1 : float
                Ending coordinates of transect line (default: spatial bounds).
            z0, z1 : float
                Vertical extent for plotting (default: model domain).
        
        Notes
        -----
        The plot shows:
        - Lithological units colored according to structural model color scheme
        - Model grid lines for reference
        - Optional node highlighting
        - Colorbar with unit names
        - Coordinate information in title
        
        The transect is created using FloPy's PlotCrossSection functionality
        and displays the 3D lithology array (lith_disv) along the specified line.
        """

        # Extract scalar coordinates from arrays to avoid dimension mismatch
        x0 = kwargs.get('x0', self.mesh.vgrid.xcellcenters[0])
        y0 = kwargs.get('y0', self.mesh.vgrid.ycellcenters[-1])
        x1 = kwargs.get('x1', self.mesh.vgrid.xcellcenters[0])
        y1 = kwargs.get('y1', self.mesh.vgrid.ycellcenters[-1])
        z0 = kwargs.get('z0', self.z0)
        z1 = kwargs.get('z1', self.z1)
    
        fig = plt.figure(figsize = figsize)
        ax = plt.subplot(111)
        #ax.set_aspect('equal')
        xsect = flopy.plot.PlotCrossSection(modelgrid=self.vgrid , line={"line": [(x0, y0),(x1, y1)]}, geographic_coords=True)
        csa = xsect.plot_array(a = self.lith_disv, cmap = self.structuralmodel.cmap, norm = self.structuralmodel.norm, alpha=0.8)
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


    def geomodel_transect_array(self, array, title, grid = True,
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
    
        fig = plt.figure(figsize = (12,4))
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
        Create contour map of the bottom elevation of a specified geological unit.
        
        Generates a plan-view contour map showing the bottom elevation surface
        of a geological unit, with data points and model boundaries overlaid.
        
        Parameters
        ----------
        spatial : Spatial
            Spatial data object containing model boundaries.
        mesh : Mesh
            Computational mesh with cell center coordinates.
        structuralmodel : StructuralModel
            Structural model containing geological data.
        unit : str
            Name of the geological unit to contour.
        contour_interval : float
            Vertical spacing between contour lines (m).
        
        Notes
        -----
        The method:
        1. Finds the geological layer index for the specified unit
        2. Interpolates bottom elevations onto a regular grid
        3. Creates filled contour plot with specified interval
        4. Overlays model boundaries and data points
        5. Adds colorbar and labels
        
        The contour levels are automatically calculated based on the
        elevation range and specified interval, rounded to neat values.
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