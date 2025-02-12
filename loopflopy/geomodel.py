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
    
class Geomodel:
    
    def __init__(self, scenario, vertgrid, z0, z1, **kwargs):     
           
        self.scenario = scenario                      
        self.vertgrid = vertgrid     
        self.z0 = z0
        self.z1 = z1
        
        for key, value in kwargs.items():
            setattr(self, key, value)    
        
#---------- FUNCTION TO EVALUATE GEO MODEL AND POPULATE HYDRAULIC PARAMETERS ------#

    def create_lith_dis_arrays(self, mesh, structuralmodel): # Takes the project parameters and model class.         
        print('   Creating Geomodel (lithology and discretisation arrays) for ', self.scenario, ' ...')
        t0 = datetime.now()
        
        self.strat_names = structuralmodel.strat_names[1:]
        self.nlg = len(self.strat_names)  
        z0, z1 = self.z0, self.z1
        
        def reshape_loop2mf(array):
            array = array.reshape((self.nlay, mesh.ncpl))
            array = np.flip(array, 0)
            return(array)

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
                    #xyz=np.array(xyz)
            
            litho = structuralmodel.model.evaluate_model(xyz)  # generates an array indicating lithology for every cell
            vf = structuralmodel.model.evaluate_model_gradient(xyz) # generates an array indicating gradient for every cell
            
            # Reshape to lay, ncpl   
            litho = np.asarray(litho)
            litho = litho.reshape((self.nlay, mesh.ncpl))
            litho = np.flip(litho, 0)
            self.lith = litho
            self.lith_disv = litho
            print('117', self.lith_disv.shape)
            
            angle1, angle2 = [], []
            for i in range(len(vf)):  
                angle1.append(find_angle1(vf[i]))
                angle2.append(find_angle2(vf[i]))
            self.angle1  = reshape_loop2mf(np.asarray(angle1))
            self.angle2  = reshape_loop2mf(np.asarray(angle2))
            
#---------- CON AND CON2  Finding geological layers bottoms ------#

        if self.vertgrid == 'con' or self.vertgrid == 'con2' : # CREATING DIS AND NPF ARRAYS
            
            nlay = int((z1 - z0)/self.res)
            dz = (z1 - z0)/nlay # actual resolution
            zc = np.arange(z0 + dz / 2, z1, dz)  # Cell centres
            
            xyz = []  
            for k in range(nlay):
                z = zc[k]
                for i in range(mesh.ncpl):    
                    x, y = mesh.xcyc[i][0], mesh.xcyc[i][1]
                    xyz.append([x,y,z])
                    
            
            litho = structuralmodel.model.evaluate_model(xyz)  # generates an array indicating lithology for every cell
            litho = np.asarray(litho)
            litho = litho.reshape((nlay, mesh.ncpl)) # Reshape to lay, ncpl
            litho = np.flip(litho, 0)
            

            def start_stop_arr(initial_list): # Function to look down pillar and pick geo bottoms
                a = np.asarray(initial_list)
                mask = np.concatenate(([True], a[1:] != a[:-1], [True]))
                idx = np.flatnonzero(mask)
                l = np.diff(idx)
                start = np.repeat(idx[:-1], l)
                stop = np.repeat(idx[1:]-1, l)
                return(start, stop)
            
            # Arrays for geo arrays
            #top = P.z1 * np.ones((mesh.ncpl), dtype=float)
            top_geo     = 9999 * np.ones((mesh.ncpl), dtype=float) # empty array to fill in ground surface
            botm_geo    = np.zeros((self.nlg, mesh.ncpl), dtype=float) # bottom elevation of each geological layer
            thick_geo   = np.zeros((self.nlg, mesh.ncpl), dtype=float) # geo layer thickness
            idomain_geo = np.zeros((self.nlg, mesh.ncpl), dtype=float)      # idomain array for each lithology
                       
            # ------BOTM_GEO ---------------
            for icpl in range(mesh.ncpl): 
                #get strat column
                strat_log = litho[:,icpl]
                present = np.unique(strat_log)
                start, stop =  start_stop_arr(strat_log)
                start = np.unique(start)
                stop = np.unique(stop)
                if icpl == 2635:
                    print(strat_log)
                    print(present)
                    print(start)
                    print(stop)
                for i, lith in enumerate(present):           
                    if lith < 0:
                        top_geo[icpl] = z1 - (stop[i]+1) * dz     
                        if icpl == 2635:
                            print('lith = ', lith)
                            print(' top_geo = ', z1 - (stop[i]+1) * dz)
                    if lith >= 0:
                        idomain_geo[lith, icpl] = 1
                        botm_geo[lith, icpl] = z1 - (stop[i]+1) * dz
                        if icpl == 2635:
                            print('lith = ', lith)
                            print(' botm_geo = ', z1 - (stop[i]+1) * dz)
                for lay_geo in range(self.nlg):
                    if idomain_geo[lay_geo, icpl] == 0: # if pinched out geological layer...
                        if lay_geo == 0:
                            botm_geo[lay_geo, icpl] = top_geo[icpl]  
                        else:
                            botm_geo[lay_geo, icpl] = botm_geo[lay_geo-1, icpl]  
            # -------------------------------
            
            for lay_geo in range(self.nlg):
                    if lay_geo == 0:
                        thick_geo[lay_geo, :] = top_geo - botm_geo[lay_geo,:]
                    else:
                        thick_geo[lay_geo, :] = botm_geo[lay_geo-1,:] - botm_geo[lay_geo,:]
                        
            self.top_geo = top_geo
            self.botm_geo = botm_geo  
            self.thick_geo = thick_geo    
            self.idomain_geo = idomain_geo 
                    
            
#----- CON - CREATE LITH, BOTM AND IDOMAIN ARRAYS (PILLAR METHOD, PICKS UP PINCHED OUT LAYERS) ------#    
        if self.vertgrid == 'con':
            self.nlay   = self.nlg * self.nls # number of model layers = geo layers * sublayers 
            botm        = np.zeros((self.nlay, mesh.ncpl), dtype=float) # bottom elevation of each model layer
            idomain     = np.ones((self.nlay, mesh.ncpl), dtype=int)    # idomain for each model layer

            for icpl in range(mesh.ncpl): 
                for lay_geo in range(self.nlg):
                    for lay_sub in range(self.nls):
                        lay = lay_geo * self.nls + lay_sub
                        if idomain_geo[lay_geo, icpl] == 0: # if pinched out geological layer...
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
            print('con 2, line 312', self.lith.shape)
            self.lith_disv = lith
            print('316', self.lith_disv.shape)
            self.nlay = nlay
            
        if self.vertgrid == 'con' or self.vertgrid == 'con2' :
            # Get gradients by reevaluationg vector field at each cell
            xyz = []                         
            for lay in range(self.nlay-1, -1, -1):
                for icpl in range(mesh.ncpl):  
                    x, y, z = mesh.xcyc[icpl][0], mesh.xcyc[icpl][1], self.botm[lay, icpl] 
                    xyz.append([x,y,z])
                    #xyz=np.array(xyz)
            vf = structuralmodel.model.evaluate_model_gradient(xyz) # generates an array indicating gradient for every cell
            
            angle1, angle2 = [], []
            for i in range(len(vf)):  
                angle1.append(find_angle1(vf[i]))
                angle2.append(find_angle2(vf[i]))
            
            self.angle1  = reshape_loop2mf(np.asarray(angle1))
            self.angle2  = reshape_loop2mf(np.asarray(angle2))
            
        self.nnodes_div = len(self.botm.flatten())   
        t1 = datetime.now()
        run_time = t1 - t0
        print('Time taken = ', run_time.total_seconds())

################## PROP ARRAYS TO BE SAVED IN DISU FORMAT ##################        
    def fill_cell_properties(self, mesh): # Uses lithology codes to populate arrays 

        
        print('   Filling cell properties for ', self.scenario, ' ...')
        t0 = datetime.now()
        
        # First create an array for cellids in layered version  (before we pop cells that are absent)
        self.cellid_disv = np.empty_like(self.lith, dtype = int)
        print('357', self.lith_disv.shape)
        print('357', self.cellid_disv.shape)
        self.cellid_disu = -1 * np.ones_like(self.lith, dtype = int)
        i = 0
        for lay in range(self.nlay):
            for icpl in range(mesh.ncpl):
                self.cellid_disv[lay, icpl] = lay * mesh.ncpl + icpl
                if self.idomain[lay, icpl] != -1:
                    self.cellid_disu[lay, icpl] = i
                    i += 1
        self.ncell_disv = self.cellid_disv.size
        self.ncell_disu = np.count_nonzero(self.cellid_disu != -1)
        
#---------- PROP ARRAYS (VOX and CON) -----   
        self.k11    = np.empty_like(self.lith, dtype = float)
        self.k22    = np.empty_like(self.lith, dtype = float)
        self.k33    = np.empty_like(self.lith, dtype = float)
        self.ss     = np.empty_like(self.lith, dtype = float)
        self.sy     = np.empty_like(self.lith, dtype = float)

        for n in range(self.nlg):  # replace lithologies with parameters
            self.k11[self.lith==n] = self.hk_perlay[n] 
            self.k22[self.lith==n] = self.hk_perlay[n] 
            self.k33[self.lith==n] = self.vk_perlay[n] 
            self.ss[self.lith==n]  = self.ss_perlay[n]
            self.sy[self.lith==n]  = self.sy_perlay[n]
                   
        # Force all K tensor angles in fault zone to 0 (Loop can't calculate angles in faulted area properly yet!)
        if 'spatial.fault_poly' in globals(): #if hassattr(P,"fault_poly"):
            for icpl in range(mesh.ncpl):
                point = Point(mesh.xcyc[icpl])
                if spatial.fault_poly.contains(point):
                    for lay in range(self.nlay):
                        self.angle1[lay,icpl] = 0  
                        self.angle2[lay,icpl] = 0   
        ######################################
        
        self.lith   = self.lith[self.cellid_disu != -1].flatten()
        self.k11    = self.k11[self.cellid_disu != -1].flatten()
        self.k22    = self.k22[self.cellid_disu != -1].flatten()
        self.k33    = self.k33[self.cellid_disu != -1].flatten()
        self.ss     = self.ss[self.cellid_disu != -1].flatten()
        self.sy     = self.sy[self.cellid_disu != -1].flatten()
        self.angle1 = self.angle1[self.cellid_disu != -1].flatten()
        self.angle2 = self.angle2[self.cellid_disu != -1].flatten()
        self.angle3 = np.zeros_like(self.angle1, dtype = float)  # Angle 3 always at 0
        
        self.logk11    = logfunc(self.k11)
        self.logk22    = logfunc(self.k22)
        self.logk33    = logfunc(self.k33)
        
        run_time = datetime.now() - t0
        print('Time taken = ', run_time.total_seconds())

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

    