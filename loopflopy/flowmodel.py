import numpy as np
from datetime import datetime
import flopy
import math
import sys
import os
import matplotlib.pyplot as plt

class Flowmodel:
    
    def __init__(self, scenario, project, data, observations, mesh, geomodel, *args, **kwargs):     

        self.scenario = scenario
        self.project = project
        self.data = data
        self.mesh = mesh
        self.geomodel = geomodel
        self.observations = observations

        self.xt3d = True
        self.staggered = True
        self.newtonoptions = ['UNDER_RELAXATION']
        
    def write_flowmodel(self, transient = False, **kwargs):
        
        print('   Writing simulation and gwf for ', self.scenario, ' ...')
        t0 = datetime.now()
       
        for key, value in kwargs.items():
            setattr(self, key, value)    

        # -------------- SIM -------------------------
        sim = flopy.mf6.MFSimulation(sim_name = 'sim', 
                                     version = 'mf6',
                                     exe_name = self.project.mfexe_name, 
                                     sim_ws = self.project.workspace)

        # -------------- TDIS -------------------------
      
        if not transient: 
            tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim)      
            
        if transient:  
            tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim, nper=len(self.perioddata), perioddata=self.perioddata)
        
        # -------------- IMS -------------------------
        # Make linear solver (inner) an order of magnitude tighter than non-linear solver (outer)
        if not transient: 
            ims = flopy.mf6.ModflowIms(sim, print_option='SUMMARY', 
                                       complexity    = 'Moderate',
                                       outer_dvclose = 1e-2, 
                                       inner_dvclose = 1e-3, 
                                       outer_maximum = 400, 
                                       linear_acceleration = "BICGSTAB",
                                       preconditioner_levels=5, #1 to 5... PLAY WITH THIS FOR SPEED UP!
                                       preconditioner_drop_tolerance=0.01, # ...if fill 7-18 (hard), DT 1e-2 (7) to 1e-5 (18)
                                       number_orthogonalizations=2,
                                      )
        if transient: 
            ims = flopy.mf6.ModflowIms(sim, print_option='SUMMARY', 
                                       complexity    = 'Moderate',
                                       outer_dvclose = 1e-2, 
                                       inner_dvclose = 1e-3, 
                                       outer_maximum = 60, 
                                       linear_acceleration = "BICGSTAB",
                                       preconditioner_levels=5, #1 to 5... PLAY WITH THIS FOR SPEED UP!
                                       preconditioner_drop_tolerance=0.01, # ...if fill 7-18 (hard), DT 1e-2 (7) to 1e-5 (18)
                                       number_orthogonalizations=2, # NORTH - increase if hard!
                                       )

        # -------------- GWF -------------------------
        gwf = flopy.mf6.ModflowGwf(sim, 
                                   modelname=self.scenario, 
                                   save_flows=True, 
                                   newtonoptions = self.newtonoptions,) 

        # -------------- DIS -------------------------       

        import disv2disu
        #from importlib import reload
        #reload(disv2disu)
        Disv2Disu = disv2disu.Disv2Disu           
         
        dv2d = Disv2Disu(self.mesh.vertices, self.mesh.cell2d, self.geomodel.top_geo, self.geomodel.botm, staggered=self.staggered, disv_idomain = self.geomodel.idomain)
        disu_gridprops = dv2d.get_gridprops_disu6()
        disu = flopy.mf6.ModflowGwfdisu(gwf, **disu_gridprops) # This is the flow package
        
        # -------------- NPF -------------------------
               
        npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(gwf, 
                                                       xt3doptions = self.xt3d, 
                                                       k = self.geomodel.k11, 
                                                       k22  = self.geomodel.k22, 
                                                       k33  = self.geomodel.k33, 
                                                       angle1 = self.geomodel.angle1, 
                                                       angle2 = self.geomodel.angle2,
                                                       angle3 = self.geomodel.angle3, 
                                                       #angle1 = 0., angle2 = 0., angle3 = 0.,
                                                       icelltype = 1,
                                                       save_flows = True, 
                                                       save_specific_discharge = True,)
                                                       #dev_minimum_saturated_thickness = 1)# try 0.1 then 0.001... no more than 1m!
        
        # -------------- IC -------------------------
        ic = flopy.mf6.ModflowGwfic(gwf, strt = self.data.strt)

         # -------------- WEL / STO -------------------------
    
        if transient and self.wel:  
            print('coo coo')
            sto = flopy.mf6.modflow.mfgwfsto.ModflowGwfsto(gwf, 
                                                           storagecoefficient=None, 
                                                           iconvert=1, 
                                                           ss = self.geomodel.ss, 
                                                           sy = self.geomodel.sy)
            wel = flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(gwf, 
                                                           print_input=True, 
                                                           print_flows=True, 
                                                           stress_period_data = self.data.spd_wel, 
                                                           save_flows=True,) 
              
        # -------------- CHD-------------------------
        if self.chd:
            
            chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(gwf, 
                                                           maxbound = len(self.data.chd_rec),
                                                           stress_period_data = self.data.chd_rec,)
        # -------------- RCH-------------------------
        if self.rch:
            rch = flopy.mf6.modflow.mfgwfrch.ModflowGwfrch(gwf, 
                                                           maxbound = len(self.data.rch_rec),
                                                           stress_period_data = self.data.rch_rec,)          
        # -------------- OBS -------------------------
        if self.obs: 
            csv_file = self.scenario + "_observations.csv" # To write observation to
            obs = flopy.mf6.ModflowUtlobs(gwf, 
                                          filename=self.scenario, 
                                          print_input=True, 
                                          continuous={csv_file: self.observations.obs_rec},) 
        # ------------ OC ---------------------------------
        oc = flopy.mf6.ModflowGwfoc(gwf, 
                                    budget_filerecord='{}.bud'.format(self.scenario), 
                                    head_filerecord='{}.hds'.format(self.scenario),
                                    saverecord=[('HEAD', 'LAST'),('BUDGET', 'ALL')], 
                                    printrecord=None,)
        
        # -------------- WRITE SIMULATION -------------------------
            
        sim.write_simulation(silent = True)   
        run_time = datetime.now() - t0
        print('   Time taken to write flow model = ', run_time.total_seconds())

         # --------------------------------------------------------
        return(sim)

    def run_flowmodel(self, sim, transient = False):
    
        t0 = datetime.now()
        print('Running simulation for ', self.scenario, ' ...')

        success, buff = sim.run_simulation(silent = True)   
        print('Model success = ', success)
        run_time = datetime.now() - t0
        print('   run_time = ', run_time.total_seconds())
        
        def process_results():
                        
            gwf = sim.get_model(self.scenario)
            package_list = gwf.get_package_list()
            print(package_list)
            times = gwf.output.head().get_times()
            head = gwf.output.head().get_data()[-1] # last time
            print('head results shape ', head.shape)
            
            bud = gwf.output.budget()
            spd = bud.get_data(text='DATA-SPDIS')[0]
            chdflow = bud.get_data(text='CHD')[-1]
            obs_data = gwf.obs

            self.gwf = gwf
            self.head = head
            self.obsdata = obs_data
            self.spd = spd
            self.chdflow = chdflow
            self.runtime = run_time.total_seconds()
            
        if success:
            process_results()

        else:
            print('   Re-writing IMS - Take 2')
            sim.remove_package(package_name='ims')
            
            ims = flopy.mf6.ModflowIms(sim, print_option='ALL', 
                                   complexity    = 'Complex',
                                   outer_dvclose = 1e-4, 
                                   inner_dvclose = 1e-6, 
                                   outer_maximum = 60,
                                   inner_maximum = 60,
                                   linear_acceleration = "BICGSTAB",
                                   backtracking_number = 10,
                                   backtracking_tolerance = 100, #1.01 (aggressive) to 10000
                                   backtracking_reduction_factor = 0.5, # 0.1-0.3, or 0.9 when non-linear convergence HARD 
                                   #preconditioner_levels=10, #1 to 5... PLAY WITH THIS FOR SPEED UP!
                                   #preconditioner_drop_tolerance=0.01, # ...if fill 7-18 (hard), DT 1e-2 (7) to 1e-5 (18)
                                   #number_orthogonalizations=10,
                                   ) # NORTH - increase if hard!)

            sim.ims.write()
            success2, buff = sim.run_simulation(silent = True)   
            print('Model success2 = ', success2)
            
            if success2:
                process_results()
     
            
            else:
                print('   Re-writing IMS - Take 3')
                
                if transient:   # Increase number of timesteps to help convergence
                    future_years = 5
                    nts_future = future_years * 12
                    tdis_future = [(future_years * 365, nts_future, 1.2)] # period length, number of timesteps, tsmult
                    sim.remove_package(package_name='tdis')
                    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(sim, nper=len(tdis_future), perioddata=tdis_future)
                
                # More aggressive solver settings
                sim.remove_package(package_name='ims')
                ims = flopy.mf6.ModflowIms(sim, print_option='ALL', 
                            complexity    = 'Complex',
                            outer_dvclose = 1e-4, 
                            inner_dvclose = 1e-6, 
                            outer_maximum = 60,
                            inner_maximum = 300,
                            linear_acceleration = "BICGSTAB",
                            reordering_method=['RCM'],
                            #no_ptcrecord = ['ALL'],
                            under_relaxation = 'DBD',
                            under_relaxation_kappa = 0.2, #0.05 (aggressive) to 0.3
                            under_relaxation_theta = 0.7, # 0.5 - 0.9
                            under_relaxation_gamma = 0.1, # 0-0.2 doesnt make big difference
                            under_relaxation_momentum = 0.001, #0-0.001 doesn't make big difference
                            backtracking_number = 15,
                            backtracking_tolerance = 1.1, #1.01 (aggressive) to 10000
                            backtracking_reduction_factor = 0.7, # 0.1-0.3, or 0.9 when non-linear convergence HARD 
                            preconditioner_levels=18, #1 to 5... PLAY WITH THIS FOR SPEED UP!
                            preconditioner_drop_tolerance=0.00001, # ...if fill 7-18 (hard), DT 1e-2 (7) to 1e-5 (18)
                            number_orthogonalizations=10,)
                sim.ims.write()
                success3, buff = sim.run_simulation(silent = True)   
                print('Model success3 = ', success3)
                
                if success3:
                    process_results()

    def get_watertable(self, geomodel, heads, hdry=-1e30):
        nlay, ncpl = geomodel.cellid_disv.shape
    
        head_disv = -999 * np.ones((geomodel.ncell_disv))  
        watertable = -999 * np.ones(ncpl)
    
        for disucell in range(geomodel.ncell_disu):
            disvcell = np.where(geomodel.cellid_disu.flatten()==disucell)[0][0]  
            head_disv[disvcell] = heads[0][disucell]   
        head_disv = head_disv.reshape((nlay, ncpl))
    
        for icpl in range(ncpl):
            for lay in range(nlay): 
                if geomodel.idomain[lay, icpl] == 1:    # if present
                    if head_disv[lay, icpl] != hdry:    # if not dry
                        watertable[icpl] = head_disv[lay, icpl] 
                        break           
    
        return watertable

    def plot_watertable(self, spatial, mesh, geomodel, flowmodel, watertable, extent = None, vmin = None, vmax = None):
        fig = plt.figure(figsize = (8,6))
        ax = plt.subplot(111)
        ax.set_title(flowmodel.scenario, size = 10)
        mapview = flopy.plot.PlotMapView(modelgrid=geomodel.vgrid)#, layer = layer)
        plan = mapview.plot_array(watertable, cmap='Spectral', alpha=0.8, vmin = vmin, vmax = vmax)
        #if vectors:
        #    mapview.plot_vector(flowmodel.spd["qx"], flowmodel.spd["qy"], alpha=0.5)
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('y (m)', size = 10)
        plt.colorbar(plan, shrink = 0.4)
           
        for j in range(spatial.nobs):
            ax.plot(spatial.xyobsbores[j][0], spatial.xyobsbores[j][1],'o', ms = '4', c = 'black')
            #ax.annotate(spatial.idobsbores[j], (spatial.xyobsbores[j][0], spatial.xyobsbores[j][1]+100), c='black', size = 10) #, weight = 'bold')
        
        for j in range(spatial.npump):
            ax.plot(spatial.xypumpbores[j][0], spatial.xypumpbores[j][1],'o', ms = '4', c = 'red')
            #ax.annotate(spatial.idpumpbores[j], (spatial.xypumpbores[j][0], spatial.xypumpbores[j][1]+100), c='red', size = 10) #, weight = 'bold')
            
        if mesh.plangrid == 'car': mesh.sg.plot(color = 'black', lw = 0.2) 
        if mesh.plangrid == 'tri': mesh.tri.plot(edgecolor='black', lw = 0.2)
        if mesh.plangrid == 'vor': mesh.vor.plot(edgecolor='black', lw = 0.2)
        ax.plot([extent[0], extent[1]], [extent[2], extent[3]], color = 'black', lw = 1)
        plt.tight_layout() 

    def plot_plan(self, spatial, mesh, array, layer, extent = None, vmin = None, vmax = None, vectors = None):
        
        fig = plt.figure(figsize = (8,6))
        ax = plt.subplot(111)
        ax.set_title(self.scenario, size = 10)
        mapview = flopy.plot.PlotMapView(model=self.gwf)#, layer = layer)
        plan = mapview.plot_array(getattr(self, array), cmap='Spectral', alpha=0.8, vmin = vmin, vmax = vmax)
        if vectors:
            mapview.plot_vector(self.spd["qx"], self.spd["qy"], alpha=0.5)
        ax.set_xlabel('x (m)', size = 10)
        ax.set_ylabel('y (m)', size = 10)
        plt.colorbar(plan, shrink = 0.4)
           
        for j in range(spatial.nobs):
            ax.plot(spatial.xyobsbores[j][0], spatial.xyobsbores[j][1],'o', ms = '4', c = 'black')
            #ax.annotate(spatial.idobsbores[j], (spatial.xyobsbores[j][0], spatial.xyobsbores[j][1]+100), c='black', size = 10) #, weight = 'bold')
        
        for j in range(spatial.npump):
            ax.plot(spatial.xypumpbores[j][0], spatial.xypumpbores[j][1],'o', ms = '4', c = 'red')
            #ax.annotate(spatial.idpumpbores[j], (spatial.xypumpbores[j][0], spatial.xypumpbores[j][1]+100), c='red', size = 10) #, weight = 'bold')
            
        if mesh.plangrid == 'car': mesh.sg.plot(ax = ax, color = 'black', lw = 0.2) 
        if mesh.plangrid == 'tri': mesh.tri.plot(ax = ax, edgecolor='black', lw = 0.2)
        if mesh.plangrid == 'vor': mesh.vor.plot(ax = ax, edgecolor='black', lw = 0.2)
        ax.plot([extent[0], extent[1]], [extent[2], extent[3]], color = 'black', lw = 1)
        plt.tight_layout() 

