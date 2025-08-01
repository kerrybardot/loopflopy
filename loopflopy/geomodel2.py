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

logfunc = lambda e: np.log10(e)

# angle 1 (DIP DIRECTION) rotates around z axis counterclockwise looking from +ve z.
def find_angle1(nv): # nv = normal vector to surface
    # The dot product of perpencicular vectors = 0
    # A vector perpendicular to nv would be [a,b,c]
    import numpy as np
    import math
    if nv[2] == 0:
        angle1 = 0.
    else:
        a = nv[0]
        b = nv[1]
        c = -(a*nv[0]+b*nv[1])/nv[2]
        v = [a,b,c]
        if np.isnan(v[0]) == True or np.isnan(v[1]) == True: 
            angle1 = 0.
        if v[0] == 0.:
            if v[1] > 0:
                angle1 = 90
            else:
                angle1 = -90
        else:             
            tantheta = v[1]/v[0] 
            angle1 = np.degrees(math.atan(tantheta))
    return(angle1)

# angle 2 (DIP) rotates around y axis clockwise looking from +ve y.
def find_angle2(nv): # nv = normal vector to surface
    # The dot product of perpencicular vectors = 0
    # A vector perpendicular to nv would be [a,b,c]
    import numpy as np
    import math
    if nv[2] == 0:
        angle2 = 0.
    else:
        a = nv[0]
        b = nv[1]
        c = -(a*nv[0]+b*nv[1])/nv[2]
        v = [a,b,c]
        if np.isnan(v[0]) == True or np.isnan(v[1]) == True or np.isnan(v[2]) == True:
            angle2 = 0.
        else:
            v_mag = (v[0]**2 + v[1]**2 + v[2]**2)**0.5 
            costheta = v[2]/v_mag
            angle2 = 90-np.degrees(math.acos(costheta)) 
    return(angle2)

def reshape_loop2mf(array, nlay, ncpl):
    array = array.reshape((nlay, ncpl))
    array = np.flip(array, 0)
    return(array)
    
class Geomodel:
    
    def __init__(self, scenario, vertgrid, z0, z1, transect = False, nlg = None, **kwargs):     
           
        self.scenario = scenario                      
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

    def create_model_layers(self, mesh, structuralmodel, dem):

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
                    print(np.unique(surface, return_counts=True))

                    surfaces.append(surface)
                return surfaces

            surfaces = make_surfaces(structuralmodel, mesh) # round surface is surface 0!

            top_geo = dem.topo # Make top of geomodel equal to DEM topography

            if self.nlg == len(structuralmodel.lithids)-1: # Situation when model bottom is z0
                for i in range(1, self.nlg): # don't include groud level or bottom surface
                    botm_geo[i-1] = surfaces[i] # Bottom of geological layer
                    botm_geo[-1] = self.z0 # Bottom of the last geological layer is the model bottom
            else: # Situation when model bottom is bottom of geo layer
                for i in range(1, self.nlg+1): # don't include groud level
                    botm_geo[i-1] = surfaces[i] # Bottom of geological layer

            print('top_geo shape', top_geo.shape)
            print('botm_geo', botm_geo.shape)
            
            # Here we make sure that the bottom of the first layer is below the ground surface
            # And if not, we make it equal to the ground surface!
            # This is a bit cheating, as if the top layer is not there, I've made it 1m thick
            # Only because I'm not sure my code can cope with pinched out layers in the top layer - to be fixed later!
            new_array = np.where(top_geo - botm_geo[0] < 0, top_geo-1, botm_geo[0]) 
            botm_geo[0] = new_array
            
            # Here we need to flag where there are pinched out cells in the top layer.
            new_array = np.where(top_geo - botm_geo[0] <= 0, 0, 1) # if the botm of the first layer is above groundsurface, make idomain 0
            idomain_geo[0] = new_array

            # Here we need to flag where there are pinched out cells for all other layers.
            for i in range(1,self.nlg):
                new_array = np.where(botm_geo[i-1] - botm_geo[i] <= 0, 0, 1)
                idomain_geo[i] = new_array # if the top of the layer is above the bottom of the layer, make idomain 0

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
    
            self.botm_geo = botm_geo      
            self.botm = botm
            self.idomain = idomain
            self.lith = lith
            self.lith_disv = lith
            self.nlay = nlay



        ###############
            
        if self.vertgrid == 'con' or self.vertgrid == 'con2' :

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


            print('\n   5. Calculating gradients...')
            t0 = datetime.now()

            if self.transect == False:
                xyz = []                         
                for lay in range(self.nlay-1, -1, -1):
                    for icpl in range(mesh.ncpl):  
                        x, y, z = mesh.xcyc[icpl][0], mesh.xcyc[icpl][1], self.botm[lay, icpl] 
                        xyz.append([x,y,z])
                vf = structuralmodel.model.evaluate_model_gradient(xyz) # generates an array indicating gradient for every cell
                
                ang1, ang2 = [], []
                for i in range(len(vf)):  
                    ang1.append(find_angle1(vf[i]))
                    ang2.append(find_angle2(vf[i]))

            else:
                vf = np.gradient(self.botm, mesh.delx) 
                #print('vf1 shape ', vf[1].shape)
                #print('vf 1 ', vf[1])
                vf = vf[1].flatten()
                print('vf shape ', vf.shape)
                ang2 = []
                for i in range(len(vf)):  # angle 2 (DIP) rotates around y axis clockwise looking from +ve y.
                    angle2 = np.degrees(math.atan(vf[i]))
                    ang2.append(angle2)
                
                ang1 = np.zeros_like(ang2, dtype = float)  # Angle 1 always at 0 in transect

            self.ang1  = reshape_loop2mf(np.asarray(ang1), self.botm.shape[0], self.botm.shape[1])
            self.ang2  = reshape_loop2mf(np.asarray(ang2), self.botm.shape[0], self.botm.shape[1])
            
        self.nnodes_div = len(self.botm.flatten())   
        self.thick = np.zeros((self.nlay, mesh.ncpl))
        self.thick[0,:] = self.top_geo - self.botm[0]
        self.thick[1:-1,:] = self.botm[0:-2] - self.botm[1:-1]
        self.thick.min()
        self.zc = self.botm + self.thick/2

        self.vgrid = flopy.discretization.VertexGrid(vertices=mesh.vertices, cell2d=mesh.cell2d, ncpl = mesh.ncpl, 
                                                     top = self.top_geo, botm = self.botm)
    
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

    def geomodel_transect_lith(self, structuralmodel, spatial, **kwargs):
        x0 = kwargs.get('x0', spatial.x0)
        y0 = kwargs.get('y0', spatial.y0)
        z0 = kwargs.get('z0', self.z0)
        x1 = kwargs.get('x1', spatial.x1)
        y1 = kwargs.get('y1', spatial.y1)
        z1 = kwargs.get('z1', self.z1)
    
        fig = plt.figure(figsize = (12,4))
        ax = plt.subplot(111)
        xsect = flopy.plot.PlotCrossSection(modelgrid=self.vgrid , line={"line": [(x0, y0),(x1, y1)]}, geographic_coords=True)
        csa = xsect.plot_array(a = self.lith_disv, cmap = structuralmodel.cmap, norm = structuralmodel.norm, alpha=0.8)
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('z (m)', size = 10)
        ax.set_ylim([z0, z1])
  
        linecollection = xsect.plot_grid(lw = 0.1, color = 'black') 
        
        labels = structuralmodel.strat_names[1:]
        ticks = [i for i in np.arange(0,len(labels))]
        boundaries = np.arange(-1,len(labels),1)+0.5

        cbar = plt.colorbar(csa,
                            boundaries = boundaries,
                            shrink = 1.0
                            )
        cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')    
        plt.title(f"x0, x1, y0, y1 = {x0:.0f}, {x1:.0f}, {y0:.0f}, {y1:.0f}", size=8)
        plt.tight_layout()  
        plt.savefig('../figures/geomodel_transect.png')
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