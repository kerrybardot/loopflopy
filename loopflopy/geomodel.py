import numpy as np
import flopy
import math
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors
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
    
    def __init__(self, scenario, vertgrid, z0, z1, transect = False, **kwargs):     
           
        self.scenario = scenario                      
        self.vertgrid = vertgrid     
        self.z0 = z0
        self.z1 = z1
        self.transect = transect
        

        for key, value in kwargs.items():
            setattr(self, key, value)    
        
#---------- FUNCTION TO EVALUATE GEO MODEL AND POPULATE HYDRAULIC PARAMETERS ------#

    def evaluate_structuralmodel(self, mesh, structuralmodel): # Takes the project parameters and model class.         
        #print('   Creating Geomodel (lithology and discretisation arrays) for ', self.scenario, ' ...')
        
        self.strat_names = structuralmodel.strat_names[1:]
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

            print('   1. Evaluating structural model...')
            t0 = datetime.now()
            self.xyz = xyz  
            print('len(xyz) = ', len(xyz))      
            litho = structuralmodel.model.evaluate_model(np.array(xyz))  # generates an array indicating lithology for every cell
            litho = np.asarray(litho)
            litho = litho.reshape((nlay, mesh.ncpl)) # Reshape to lay, ncpl
            litho = np.flip(litho, 0)
            self.litho = litho

            t1 = datetime.now()
            run_time = t1 - t0
            print('Time taken Block 1 (Evaluate model) = ', run_time.total_seconds())

    def create_model_layers(self, mesh, structuralmodel):
            
        self.units = np.array(structuralmodel.strat_names[1:])  

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
            
            # ------------------------------------------
            print('   2. Creating geo model layers...')
            t0 = datetime.now()

            # Arrays for geo arrays
            top_geo     = 9999 * np.ones((mesh.ncpl), dtype=float) # empty array to fill in ground surface
            botm_geo    = np.zeros((self.nlg, mesh.ncpl), dtype=float) # bottom elevation of each geological layer
            thick_geo   = np.zeros((self.nlg, mesh.ncpl), dtype=float) # geo layer thickness
            idomain_geo = np.zeros((self.nlg, mesh.ncpl), dtype=float)      # idomain array for each lithology
            
            stop_array = np.zeros((self.nlg+1, mesh.ncpl), dtype=float)
            print('stop_array shape', stop_array .shape )
            V = self.litho
            W = V[1:, :] - V[:-1, :] 
            #W = V[1:, :] - V[:-1, :] 
            if np.any(W < 0):
                print(f'Geology repeats itself {np.sum(W < 0)} times at {np.where(W < 0)}')
                W[W < 0] = 0 # this is to handle instances where geology goes in reverse order (back to a younger sequence)

            
            print('nlay = ', nlay)
            print('ncpl = ', mesh.ncpl)
            print('nlg number of geo layers = ', self.nlg)

            for icpl in range(mesh.ncpl):
                #print('ICPL = ', icpl)
                #if icpl == 297:
                #    print('Pillar lithologies ', V[:,icpl])
                #    print('V - V ', V[1:, icpl] - V[:-1, icpl])

                # IDOMAIN
                present = np.unique(V[:,icpl])
                for p in present:
                    if p >= 0: # don't include above ground
                        idomain_geo[p, icpl] = 1
                
                stop = np.array([nlay-1]) # Add the last layer to start with
                #print('line 225 stop ', stop)
                for i in range(1,self.nlg): 
                    
                    #idx = np.where(V[1:, icpl] - V[:-1, icpl] == i)[0] # checking if different from row above
                    idx = np.where(W[:,icpl] == i)[0] # checking if different from row above
                    # e.g. if i =  3 and idx =  [146 199], then it skips 3 layers TWICE!
                    if idx.size != 0: # If there returns an index
                        #print('i = ', i, 'idx = ', idx)
                        
                        for id in idx:
                            #if icpl == 297: print('id = ', id, 'idx = ', idx, 'np.ones(i) = ', np.ones(i))
                            idx_array = id * np.ones(i)
                            stop = np.concatenate((stop, idx_array))
    
                n = self.nlg+1 - len(stop)# number of pinched out layers at the bottom not yet added to stop array
                #print('n = ', n)
                #print('stop  ', stop)
                m = (nlay-1) * np.ones(n) # this is just accounting for the bottom geo layers that dont exist in pillar
                stop = np.concatenate((stop, m))

                stop = np.sort(stop)
                #if icpl == 297: print('stop ', stop)     
                stop_array[:,icpl] = stop


            #print('stop_array.shape = ', stop_array.shape)
            botm_geo = self.z1 - (stop_array + 1)* self.dz
            botm_geo[:, 0]
            top_geo = botm_geo[0,:]
            botm_geo = botm_geo[1:,:]
            #print('top_geo shape', top_geo.shape)
            #print('botm_geo', botm_geo.shape)
            t1 = datetime.now()
            run_time = t1 - t0
            print('Time taken Block 2 create geomodel layers ', run_time.total_seconds())

            '''for icpl in range(mesh.ncpl): 
                #get strat column
                V = litho[:,icpl] # stratigraphy id in pillar
                W = np.where(V[:-1] != V[1:])[0] # index of strat changes
                present = np.unique(strat_log)
                start, stop =  start_stop_arr(strat_log)
                start = np.unique(start)
                stop = np.unique(stop)
                if icpl == 20:
                    self.strat_log = strat_log
                    print('strat_log ', strat_log)
                    print('present ', present)
                    print('start ', start)
                    print('stop ', stop)
                for i, lith in enumerate(present):           
                    if lith < 0:
                        top_geo[icpl] = z1 - (stop[i]+1) * dz     
                        if icpl == 20:
                            print('lith = ', lith)
                            print(' top_geo = ', z1 - (stop[i]+1) * dz)
                    if lith >= 0:
                        idomain_geo[lith, icpl] = 1
                        botm_geo[lith, icpl] = z1 - (stop[i]+1) * dz
                        if icpl == 20:
                            print('lith = ', lith)
                            print('stop[i] = ', stop[i])
                            print(' botm_geo = ', z1 - (stop[i]+1) * dz)
                for lay_geo in range(self.nlg):
                    if idomain_geo[lay_geo, icpl] == 0: # if pinched out geological layer...
                        if lay_geo == 0:
                            botm_geo[lay_geo, icpl] = top_geo[icpl]  
                        else:
                            botm_geo[lay_geo, icpl] = botm_geo[lay_geo-1, icpl]  '''
                            
            # -------------------------------
            print('   3. Evaluating geo layer thicknesses...')
            t0 = datetime.now()

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
            print('Time taken Block 3 tiny bit', run_time.total_seconds())
                    
            
#----- CON - CREATE LITH, BOTM AND IDOMAIN ARRAYS (PILLAR METHOD, PICKS UP PINCHED OUT LAYERS) ------#    
        if self.vertgrid == 'con':

            print('   4. Creating flow model layers...')
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


            print('   5. Calculating gradients...')
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

        # Save model layer thicknesses as (lay, icpl) array
        self.thick = np.zeros((self.nlay, mesh.ncpl))
        self.thick[0,:] = self.top_geo - self.botm[0]
        self.thick[1:-1,:] = self.botm[0:-2] - self.botm[1:-1]
        self.thick.min()

        # Save cell centres z value as (lay, icpl) array
        self.zc = np.zeros((self.nlay, mesh.ncpl))
        for lay in range(self.nlay):
            for icpl in range(mesh.ncpl):
                self.zc[lay, icpl] = self.botm[lay, icpl] + self.thick[lay, icpl]/2

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
        for n in range(self.nlg): self.iconvert[self.lith_disv==n]  = self.iconvert_perlay[n]
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
        ax.set_title('surface geology', size = 10)

        if mesh.plangrid == 'car': mesh.sg.plot(color = 'black', lw = 0.2) 
        if mesh.plangrid == 'tri': mesh.tri.plot(edgecolor='black', lw = 0.2)
        if mesh.plangrid == 'vor': mesh.vor.plot(edgecolor='black', lw = 0.2)

        mapview = flopy.plot.PlotMapView(modelgrid=self.vgrid, layer = 0)
        
        plan = mapview.plot_array(self.surf_lith, cmap=structuralmodel.cmap, norm = structuralmodel.norm, alpha=0.8)
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('y (m)', size = 10)
        
        labels = structuralmodel.strat_names[1:]
        ticks = [i for i in np.arange(0,len(labels))]
        boundaries = np.arange(-1,len(labels),1)+0.5       
        
        ax.plot([x0, x1], [y0, y1], color = 'black', lw = 1)
        for node in spatial.interface_nodes:
            ax.plot(node[0], node[1], 'o', color = 'black', markersize = 2)
        cbar = plt.colorbar(plan,
                    boundaries = boundaries,
                    shrink = 1.0)
        cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')   
        plt.tight_layout()  
        plt.show()

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
        #linecollection = xsect.plot_grid(lw = 0.1, color = 'black') # Don't plot grid for reference
        
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

    