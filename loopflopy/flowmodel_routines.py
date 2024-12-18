#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import flopy
from shapely.geometry import LineString,Point,Polygon,shape
import math

  
def plot_watertable(spatial, mesh, geomodel, flowmodel, watertable, extent = None, vmin = None, vmax = None):
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
    
def plot_plan(spatial, mesh, flowmodel, array, layer, extent = None, vmin = None, vmax = None, vectors = None):
    
    fig = plt.figure(figsize = (8,6))
    ax = plt.subplot(111)
    ax.set_title(flowmodel.scenario, size = 10)
    mapview = flopy.plot.PlotMapView(model=flowmodel.gwf)#, layer = layer)
    plan = mapview.plot_array(getattr(flowmodel, array), cmap='Spectral', alpha=0.8, vmin = vmin, vmax = vmax)
    if vectors:
        mapview.plot_vector(flowmodel.spd["qx"], flowmodel.spd["qy"], alpha=0.5)
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
    

def plot_transect(spatial, flowmodel, array, X0 = None, X1 = None, Y0 = None, Y1 = None, vmin = None, vmax = None): 
    fig = plt.figure(figsize = (8, 3))
    ax = plt.subplot(111)
    ax.set_title(flowmodel.scenario, size = 10)
    xsect = flopy.plot.PlotCrossSection(model=flowmodel.gwf, line={"line": [(X0, Y0),(X1, Y1)]}, geographic_coords=True) #extent =  [X0, X1, Y0, Y1],)# 
    csa = xsect.plot_array(a = getattr(flowmodel, array), cmap = 'Spectral', alpha=0.8, vmin = vmin, vmax = vmax)
    ax.set_xlabel('distance (m)', size = 10)
    ax.set_ylabel('z (m)', size = 10)
    lc = xsect.plot_grid(lw = 0.1, color = 'black')
    plt.colorbar(csa, shrink = 0.7)
    plt.tight_layout()  
            

def plot_bylayer(project, models, layer, vmin = None, vmax = None):

    fig = plt.figure(figsize=(12, 8))
    nmodels = len(models)
    for i in range(nmodels):
        ax = plt.subplot(2,3,i+1, aspect="equal")
        flowmodel = models[i]
        water_table = flopy.utils.postprocessing.get_water_table(flowmodel.gwf.output.head().get_data()[-1])
        M.heads_disv = -1e30 * np.ones_like(M.idomain, dtype=float) 
        for i, h in enumerate(water_table):
            if math.isnan(h) == False: 
                M.heads_disv[M.cellid_disu==i] = h        
        pmv = flopy.plot.PlotMapView(modelgrid=geomodel.vgrid)
        H = pmv.plot_array(flowmodel.heads_disv[layer], vmin = vmin, vmax = vmax, cmap = 'Spectral', alpha = 0.6)
        for j in range(len(spatial.xyobsbores)):
            ax.plot(spatial.xyobsbores[j][0], spatial.xyobsbores[j][1],'o', ms = '4', c = 'black')
            ax.annotate(spatialP.idobsbores[j], (spatial.xyobsbores[j][0], spatial.xyobsbores[j][1]+100), c='black', size = 12) #, weight = 'bold')
        
        for j in range(len(spatial.xypumpbores)):
            ax.plot(spatial.xypumpbores[j][0], spatial.xypumpbores[j][1],'o', ms = '4', c = 'red')
            ax.annotate(spatial.idpumpbores[j], (spatial.xypumpbores[j][0], spatial.xypumpbores[j][1]+100), c='red', size = 12) #, weight = 'bold')
            
        if mesh.plangrid == 'car': mesh.sg.plot(color = 'gray', lw = 0.4) 
        if mesh.plangrid == 'tri': mesh.tri.plot(edgecolor='gray', lw = 0.4)
        if mesh.plangrid == 'vor': mesh.vor.plot(edgecolor='black', lw = 0.4)
        ax.set_title(flowmodel.scenario, size = 10)
        plt.colorbar(H, shrink = 0.4)



'''

def multiplot_vgrid_transect(P, models, array, X0, Y0, X1, Y1, vmin = None, vmax = None): # array needs to be a string of a property eg. 'k11', 'angle2'
    nmodels = len(models)
    if nmodels > 1: fig = plt.figure(figsize = (10,2*nmodels))
    if nmodels ==1: fig = plt.figure(figsize = (10,2.25))
    fig.suptitle("TRANSECT - " + array)
    for i in range(nmodels):
        M = models[i]
        a = getattr(M, array)
        ax = plt.subplot(nmodels, 1, i+1)
        ax.set_title(M.modelname, size = 10) 
        xsect = flopy.plot.PlotCrossSection(modelgrid=M.vgrid, line={"line": [(X0, Y0),(X1, Y1)]}, 
                                            extent = [P.x0,P.x1,P.z0,P.z1], geographic_coords=True)
        csa = xsect.plot_array(a = a, cmap = 'Spectral', alpha=0.8, vmin = vmin, vmax = vmax)
        if i == nmodels-1: ax.set_xlabel('x (m)', size = 10)
        if i == int(nmodels/2): ax.set_ylabel('z (m)', size = 10)
        if nmodels>1: linecollection = xsect.plot_grid(lw = 0.1, color = 'black') # Don't plot grid for reference
        plt.colorbar(csa, shrink = 0.7)
    plt.tight_layout()  
    plt.show()   


## PLOTTING HYDRAULIC PROPERTIES

def multiplot_prop_plan(P, models, array, layer, vmin = None, vmax = None):   # array needs to be a string of a property eg. 'k11', 'logk11'  
    fig = plt.figure(figsize = (10,12))
    fig.suptitle("PLAN - " + array)
    nmodels = len(models)   
    for i in range(nmodels):
        ax = plt.subplot(3,2,i+1)
        M = models[i]
        model = M.gwf
        a = getattr(M, array)
                 
        ax.set_title(M.modelname, size = 10)
        mapview = flopy.plot.PlotMapView(model=model, layer = layer)
        plan = mapview.plot_array(a, cmap='Spectral', alpha=0.8, vmin = vmin, vmax = vmax)
        linecollection = mapview.plot_grid(lw = 0.1, color = 'black')
        if i == 4 or i == 5: ax.set_xlabel('x (m)', size = 10)
        if i == 0 or i == 2 or i == 4: ax.set_ylabel('y (m)', size = 10)

        plt.colorbar(plan, shrink = 0.4)
    plt.tight_layout()  
    
def multiplot_prop_transect(P, models, array, X0, Y0, X1, Y1, vmin = None, vmax = None): # array needs to be a string of a property eg. 'k11', 'angle2'
    nmodels = len(models)
    if nmodels > 1: fig = plt.figure(figsize = (10,2*nmodels))
    if nmodels ==1: fig = plt.figure(figsize = (10,2.5))
    fig.suptitle("TRANSECT - " + array)
    for i in range(nmodels):
        M = models[i]
        model = M.gwf
        a = getattr(M, array)
        
        ax = plt.subplot(nmodels, 1, i+1)
        ax.set_title(M.modelname, size = 10) 
        xsect = flopy.plot.PlotCrossSection(model=model, line={"line": [(X0, Y0),(X1, Y1)]}, 
                                            extent = [P.x0,P.x1,P.z0,P.z1], geographic_coords=True)
        csa = xsect.plot_array(a = a, cmap = 'Spectral', alpha=0.8, vmin = vmin, vmax = vmax)
        linecollection = xsect.plot_grid(lw = 0.1, color = 'black')
        if i == nmodels-1: ax.set_xlabel('x (m)', size = 10)
        if i == int(nmodels/2): ax.set_ylabel('z (m)', size = 10)
        if nmodels>1: linecollection = xsect.plot_grid(lw = 0.1, color = 'black') # Don't plot grid for reference
        plt.colorbar(csa, shrink = 0.7)
    plt.tight_layout()  
    plt.show()    
    
### PLOTTING HEADS

def multiplot_watertable(P, models, period): 
    nmodels = len(models)
    fig = plt.figure(figsize = (10,12))
    #contours = np.arange(0, 60, 5)
    from flopy.plot import styles
    
    fig.suptitle("PLAN")
    for i in range(nmodels):
        M = models[i]
        model = M.gwf
        ax = plt.subplot(3, 2, i+1)
        ax.set_title(M.modelname, size = 10) 
        if period == 'Steady' : water_table = flopy.utils.postprocessing.get_water_table(M.head_ss, hdry=-1e30)  
        if period == 'Past'   : water_table = flopy.utils.postprocessing.get_water_table(M.head_present, hdry=-1e30)  
        if period == 'Future' : water_table = flopy.utils.postprocessing.get_water_table(M.head_future, hdry=-1e30)  
            
        m = flopy.modflow.Modflow.load(str(M.modelname + '.nam'), model_ws=P.workspace)
        pmv = flopy.plot.PlotMapView(modelgrid = m.modelgrid, ax=ax)
        #pmv.plot_vector(M.ss_spdis["qx"], M.spdis["qy"], alpha=0.5)
        linecollection = pmv.plot_grid(lw = 0.1)
        h = pmv.plot_array(water_table, cmap='Spectral')#, vmin=hmin, vmax=hmax, )    
        #if period == 'Steady' : water_table = M.head_ss
        #if period == 'Past'   : water_table = M.head_present
        #if period == 'Future' : water_table = M.head_future  
        #hmin, hmax = -10,60 #water_table.min(), water_table.max()
        
        for j in range(len(P.xyobsbores)):
            ax.plot(P.xyobsbores[j][0], P.xyobsbores[j][1],'o', ms = '4', c = 'black')
            ax.annotate(j, (P.xyobsbores[j][0], P.xyobsbores[j][1]+60), c = 'black', size = 12) #, weight = 'bold')
        
        for j in range(len(P.xypumpbores)):
            ax.plot(P.xypumpbores[j][0], P.xypumpbores[j][1],'o', ms = '4', c = 'red')
            ax.annotate(j, (P.xypumpbores[j][0], P.xypumpbores[j][1]+60), c = 'red', size = 12) #, weight = 'bold')
            
        if i == 2 or i == 3: ax.set_xlabel('x (m)', size = 10)
        if i == 0 or i == 2: ax.set_ylabel('y (m)', size = 10)
        plt.plot([P.fx1, P.fx2],[P.fy1, P.fy2], c = 'black', lw = 0.5)
        
        with styles.USGSMap():
            pmv = flopy.plot.PlotMapView(modelgrid = model.modelgrid, ax=ax)
            #pmv.plot_vector(M.ss_spdis["qx"], M.spdis["qy"], alpha=0.5)
            linecollection = pmv.plot_grid(lw = 0.1)
            h = pmv.plot_array(water_table, cmap='Spectral')#, vmin=hmin, vmax=hmax, )
            #c = pmv.contour_array(water_table, levels=contours, colors="black", linewidths=0.75, linestyles=":", )
            #plt.clabel(c, fontsize=8)
            pmv.plot_inactive()
            plt.colorbar(h, ax=ax, shrink=0.5)
        
        #pmv.plot_vector(M.ss_spdis["qx"], M.spdis["qy"], alpha=0.5)
        linecollection = pmv.plot_grid(lw = 0.1)
        
    plt.tight_layout()  '''

'''def plot_runtime_complexity():   
    titles = ['Steady', 'Past', 'Future']
    fig = plt.figure(figsize = (10,4))
    fig.suptitle('Model run times')
        
    for i in range(3): # each time period
        ax = plt.subplot(1, 3, i+1)
        ax.set_title(titles[i], size = 10)
        
        for m in range(4): 
            for n in range(nruns):
                for c in range(len(complex_options)): 
                    ax.plot(m, run_time_results[m, i, 0, n],'o', ms = '4', alpha = 0.6, c = 'green') # moderate
                    ax.plot(m, run_time_results[m, i, 1, n],'o', ms = '4', alpha = 0.6, c = 'blue')  # complex
        ax.set_ylim(0, 70)
        if i ==0: ax.set_ylabel('run_time (s)', size = 10)
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(['SS', 'US', 'SU', 'UU'])
    plt.legend(['Moderate', 'Complex'])
    plt.tight_layout()  
    fig.savefig('../figures/complexity_runtime.tif', dpi=300)'''

'''def make_vtk(P, nam_file): # from https://flopy.readthedocs.io/en/latest/Notebooks/export_vtk_tutorial.html
    from flopy.export import vtk
    from pathlib import Path
    from tempfile import TemporaryDirectory
    workspace = P.workspace
    
    ml = flopy.modflow.Modflow.load(nam_file, model_ws=workspace, check=False)

    tempdir = TemporaryDirectory()
    workspace = Path(tempdir.name)

    output_dir = P.workspace / "arrays_test"
    output_dir.mkdir(exist_ok=True)
    
    ml.dis.top.export(output_dir / "TOP", fmt="vtk")
    ml.dis.botm.export(model_bottom_dir = output_dir / "BOTM", fmt="vtk")
    ml.rch.rech.export(output_dir / "RECH", fmt="vtk", pvd=True)
    ml.upw.hk.export(model_hk_dir = output_dir / "HK", smooth=True, fmt="vtk", name="HK", point_scalars=True)
    
    # set up package export folder
    output_dir = workspace / "package_output_test"
    output_dir.mkdir(exist_ok=True)

    # export
    ml.dis.export(output_dir / "DIS", fmt="vtk")
    ml.upw.export(output_dir / "UPW", fmt="vtk", point_scalars=True, xml=True)
    ml.export(workspace / "model_output_test", fmt="vtk")
    
    # create a binary XML VTK object and enable PVD file writing
    vtkobj = vtk.Vtk(ml, xml=True, pvd=True, vertical_exageration=10)
    vtkobj = vtk.Vtk(ml, vertical_exageration=10) # Create a vtk object

    ## create some random array data
    r_array = np.random.random(ml.modelgrid.nnodes) * 100
    vtkobj.add_array(r_array, "random_data") ## add random data to the VTK object
    vtkobj.add_array(ml.dis.botm.array, "botm") ## add the model botom data to the VTK object
    vtkobj.write(output_dir / "Array_example" / "model.vtu") ## write the vtk object to file
    vtkobj = vtk.Vtk(ml, xml=True, pvd=True, vertical_exageration=10) # create a vtk object

    recharge = ml.rch.rech.transient_2ds ## add recharge to the VTK object
    vtkobj.add_transient_array(recharge, "recharge", masked_values=[0,],)
    vtkobj.write(output_dir / "tr_array_example" / "recharge.vtu") ## write vtk files
    vtkobj = vtk.Vtk(ml, xml=True, pvd=True, vertical_exageration=10) # create the vtk object

    spd = ml.wel.stress_period_data ## add well fluxes to the VTK object
    vtkobj.add_transient_list(spd, masked_values=[0,],)
    vtkobj.write(output_dir / "tr_list_example" / "wel_flux.vtu") ## write vtk files'''
