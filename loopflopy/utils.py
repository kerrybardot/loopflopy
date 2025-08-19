#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
import glob
import flopy
import math
import numbers
import matplotlib.image as mpimg
from math import pi

from scipy.interpolate import interp1d
import pandas as pd
import pickle
import shapefile
import matplotlib as mpl
from collections import OrderedDict
import subprocess as sup
from flopy.utils.triangle import Triangle as Triangle
import geopandas as gpd
from shapely.geometry import LineString,Point,Polygon,shape
import fiona
from datetime import datetime

# logfunc = lambda e: np.log10(e)

import pkgutil
def list_modules(package):
    package_path = package.__path__
    modules = [name for _, name, _ in pkgutil.iter_modules(package_path)]
    return modules

def print_object_details(obj):
    all_attributes_methods = dir(obj)
    methods = [attr for attr in all_attributes_methods if callable(getattr(obj, attr)) and not attr.startswith("__")]
    variables = [attr for attr in all_attributes_methods if not callable(getattr(obj, attr)) and not attr.startswith("__")]

    print('Methods:')
    for method in methods: print(method)
    print("\nVariables")
    for variable in variables: print(variable)


def find_kji(cell,nlay,nrow,ncol): #cellid is zerobased
    cellid = cell - 1
    k = math.floor(cellid/(ncol*nrow)) # Zero based
    j = math.floor((cellid - k*ncol*nrow)/ncol) # Zero based
    i = cellid - k*ncol*nrow - j*ncol
    return(k,j,i) # ZERO BASED!

def find_cellid(k,j,i,nlay,nrow,ncol): # returns zero based cell id
    return(i + j*ncol + k*ncol*nrow)

def x_to_col(x, delr):
    return(int(x/delr))

def y_to_row(y, delc):
    return(int(y/delc))

def z_to_lay(z, delz, zmax):
    return(int((zmax - z)/delz))

def lay_to_z(botm, top, lay, icpl=0):
    pillar = botm[:,icpl]
    if lay != 0:
        cell_top = pillar[lay-1]
    if lay == 0:
        cell_top = top[icpl]
    cell_bot = pillar[lay]
    dz = cell_top - cell_bot
    z = cell_top - dz/2        
    return(z)

def xyz_to_discell(x, y, x0, y1, dx, dy):
    col = int((x - x0)/dx)
    row = int((y1 - y)/dy)
    cell_coords = (dx/2 + col*dx, (y1 - dy/2) - row * dy)
    #print('Actual xy = %s %s, Cell col is %i row is %i, Cell centre is %s' %(x,y,col,row,cell_coords))
    return (col, row, cell_coords)

def disucell_to_disvcell(geomodel, disucell): # zerobased
    nlay, ncpl = geomodel.cellid_disv.shape
    disvcell = np.where(geomodel.cellid_disu.flatten()==disucell)[0][0]  
    return(disvcell)

def disvcell_to_disucell(geomodel, disvcell): # zerobased
    disucell = geomodel.cellid_disu.flatten()[disvcell]
    return disucell

def disvcell_to_layicpl(geomodel, disvcell): # zerobased
    nlay, ncpl = geomodel.cellid_disv.shape
    lay  = math.floor(disvcell/ncpl) # Zero based
    icpl = math.floor(disvcell - lay * ncpl) # Zero based
    return (lay,icpl)

def disucell_to_layicpl(geomodel, disucell): # zerobased
    nlay, ncpl = geomodel.cellid_disv.shape
    disvcell = np.where(geomodel.cellid_disu.flatten()==disucell)[0][0]  
    lay  = math.floor(disvcell/ncpl) # Zero based
    icpl = math.floor(disvcell - lay * ncpl) # Zero based
    return (lay,icpl)

def disucell_to_xyz(geomodel, disucell): # zerobased
    nlay, ncpl = geomodel.cellid_disv.shape
    disvcell = np.where(geomodel.cellid_disu.flatten()==disucell)[0][0]  
    lay  = math.floor(disvcell/ncpl) # Zero based
    icpl = math.floor(disvcell - lay * ncpl) # Zero based
    x = geomodel.vgrid.xcellcenters[icpl]
    y = geomodel.vgrid.ycellcenters[icpl]
    z = geomodel.vgrid.zcellcenters[lay, icpl]
    return(x,y,z)

def xyz_to_disucell(geomodel, x,y,z): # zerobased
    nlay, ncpl = geomodel.cellid_disv.shape
    point = Point(x,y,z)
    print(point)
    lay, icpl = geomodel.vgrid.intersect(x,y,z)
    disvcell = icpl + lay*ncpl
    disucell = geomodel.cellid_disu.flatten()[disvcell]
    return disucell

def xyz_to_disvcell(geomodel, x,y,z): # zerobased
    nlay, ncpl = geomodel.cellid_disv.shape
    point = Point(x,y,z)
    lay, icpl = geomodel.vgrid.intersect(x,y,z)
    disvcell = icpl + lay*ncpl
    return disvcell


        
# Writing and processing MODFLOW arrays

def write_input_files(gwf,modelname):

    headfile = '{}.hds'.format(modelname)
    head_filerecord = [headfile]
    budgetfile = '{}.bud'.format(modelname)
    budget_filerecord = [budgetfile]
    saverecord, printrecord = [('HEAD', 'ALL'), ('BUDGET', 'ALL')], [('HEAD', 'ALL')]
    oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(gwf, pname='oc', saverecord=saverecord, head_filerecord=head_filerecord,
                                                budget_filerecord=budget_filerecord, printrecord=printrecord)

def ch_flow(chdflow):
    flow_in, flow_out = 0., 0.
    for j in range(len(chdflow)):        
        if chdflow[j][2]>0: flow_out += chdflow[j][2]
        if chdflow[j][2]<0: flow_in  += chdflow[j][2]      
    return((flow_in, flow_out))

def get_q_disu(d2d, spd, flowja, gwf, staggered):

    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spd, gwf)
    # if cross-connections, recalculate qx taking into account overlap areas
    if staggered:
        gp = d2d.get_gridprops_disu6()
        iac = gp["iac"]
        ja = gp["ja"]
        ihc = gp["ihc"]
        topbycell = gp["top"]
        botbycell = gp["bot"]
        hwva = gp["hwva"]
        iconn = -1
        icell = -1
        for il in iac:
            icell += 1
            qxnumer = 0.
            qxdenom = 0.
            for ilnbr in range(il):
                iconn += 1
                if ihc[iconn] == 2:
                    inbr = ja[iconn]
                    if (inbr == icell):
                        continue
                    dz = min(topbycell[icell], topbycell[inbr]) - max(botbycell[icell], botbycell[inbr])
                    qxincr = flowja[iconn] / (hwva[iconn] * dz)
                    # equal weight given to each face, but could weight by distance instead
                    if (inbr < icell):
                        qxnumer += qxincr
                    else:
                        qxnumer -= qxincr
                    qxdenom += 1.
            qx[icell] = qxnumer / qxdenom

    #print(len(spd))
    qmag, qdir = [], []
    for i in range(len(spd)):
        qmag.append(np.sqrt(qx[i]**2 + qy[i]**2 + qz[i]**2))
        qdir.append(math.degrees(math.atan(qz[i]/qx[i])))      
    return(qmag,qx,qy,qz,qdir)

def plot_node(node, geomodel, structuralmodel, spatial, sim, scenario, features, **kwargs): # array needs to be a string of a property eg. 'k11', 'angle2'
    x0 = kwargs.get('x0', spatial.x0)
    y0 = kwargs.get('y0', spatial.y0)
    z0 = kwargs.get('z0', geomodel.z0)
    x1 = kwargs.get('x1', spatial.x1)
    y1 = kwargs.get('y1', spatial.y1)
    z1 = kwargs.get('z1', geomodel.z1)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)

    x, y, z = disucell_to_xyz(geomodel, node)
    
    print("Node one based = ", node + 1, "Node zero based = ", node)
    print("XYZ problem = ", x,y,z)

    gwf = sim.get_model(scenario)
    xv = gwf.modelgrid.xyzvertices[0][node]
    yv = gwf.modelgrid.xyzvertices[1][node]
    zv_top = gwf.modelgrid.xyzvertices[2][0][node]
    zv_bot = gwf.modelgrid.xyzvertices[2][1][node]
    xv, yv, zv_top, zv_bot
    print('cell width approx ', max(xv) - min(xv))
    print('cell length approx ', max(yv) - min(yv))
    print('cell thickness', zv_top - zv_bot)
    print('lay, icpl', disucell_to_layicpl(geomodel, node))

    #a = gwf.npf.k.get_data()
    a = geomodel.lith
    labels = structuralmodel.strat_names[1:]
    ticks = [i for i in np.arange(0,len(labels))]
    boundaries = np.arange(-1,len(labels),1)+0.5   

    fig = plt.figure(figsize = (10,6))
    ax = plt.subplot(211)
    ax.set_title("West-East Transect\nY =  %i" %(y))
    xsect = flopy.plot.PlotCrossSection(modelgrid=gwf.modelgrid, line={"line": [(spatial.x0, y),(spatial.x1, y)]},
                                        extent = [x0,x1,z0,z1], geographic_coords=True)


    csa = xsect.plot_array(a = a, cmap = structuralmodel.cmap, norm = structuralmodel.norm, 
                           alpha=0.8, vmin = vmin, vmax = vmax)
    
    ax.plot(x, z, 'o', color = 'red')
    ax.set_xlabel('x (m)', size = 10)
    ax.set_ylabel('z (m)', size = 10)
    linecollection = xsect.plot_grid(lw = 0.1, color = 'black') 
    
    ax = plt.subplot(212)
    ax.set_title("South-North Transect\nX = %i" %(x))
    xsect = flopy.plot.PlotCrossSection(modelgrid=gwf.modelgrid, line={"line": [(x, spatial.y0),(x, spatial.y1)]},
                                        extent = [y0,y1,z0,z1], geographic_coords=True)
    csa = xsect.plot_array(a = a, cmap = structuralmodel.cmap, norm = structuralmodel.norm,
                            alpha=0.8, vmin = vmin, vmax = vmax)
    ax.plot(y, z, 'o', color = 'red')
    ax.set_xlabel('y (m)', size = 10)
    ax.set_ylabel('z (m)', size = 10)
    linecollection = xsect.plot_grid(lw = 0.1, color = 'black') 
    plt.savefig('../figures/problem_node_%i_section.png' %(node))
    
    fig = plt.figure(figsize = (6,6))
    ax = plt.subplot(111)
    ax.set_title("Plan")
    mapview = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid, layer = 0)#, geographic_coords=True)
    plan = mapview.plot_array(a = a, cmap = structuralmodel.cmap, norm = structuralmodel.norm,
                               alpha=0.8, vmin = vmin, vmax = vmax)
    
    ax.plot(x, y, 'o', color = 'red')
    ax.set_xlabel('x (m)', size = 10)
    ax.set_ylabel('y (m)', size = 10)
    linecollection = mapview.plot_grid(lw = 0.1, color = 'black') 

    if 'fault' in features:
        spatial.faults_gdf.plot(ax=ax, color = 'red', zorder=2)
    
    cbar = plt.colorbar(plan, boundaries = boundaries, shrink = 0.5)
    cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')   
    plt.tight_layout()  
    plt.savefig('../figures/problem_node_%i_plan.png' %(node))
    plt.show()  


def plot_node_transect(node, geomodel, structuralmodel, spatial, sim, scenario, features, **kwargs): # array needs to be a string of a property eg. 'k11', 'angle2'
    x0 = kwargs.get('x0', spatial.x0)
    y0 = kwargs.get('y0', spatial.y0)
    z0 = kwargs.get('z0', geomodel.z0)
    x1 = kwargs.get('x1', spatial.x1)
    y1 = kwargs.get('y1', spatial.y1)
    z1 = kwargs.get('z1', geomodel.z1)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)

    x, y, z = disucell_to_xyz(geomodel, node)
    
    print("Node one based = ", node + 1, "Node zero based = ", node)
    print("XYZ problem = ", x,y,z)

    gwf = sim.get_model(scenario)
    xv = gwf.modelgrid.xyzvertices[0][node]
    yv = gwf.modelgrid.xyzvertices[1][node]
    zv_top = gwf.modelgrid.xyzvertices[2][0][node]
    zv_bot = gwf.modelgrid.xyzvertices[2][1][node]
    xv, yv, zv_top, zv_bot
    print('cell width approx ', max(xv) - min(xv))
    print('cell length approx ', max(yv) - min(yv))
    print('cell thickness', zv_top - zv_bot)
    print('lay, icpl', disucell_to_layicpl(geomodel, node))

    #a = gwf.npf.k.get_data()
    a = geomodel.lith
    labels = structuralmodel.strat_names[1:]
    ticks = [i for i in np.arange(0,len(labels))]
    boundaries = np.arange(-1,len(labels),1)+0.5   

    fig = plt.figure(figsize = (10,3))
    ax = plt.subplot(111)
    ax.set_title("West-East Transect\nY =  %i" %(y))
    xsect = flopy.plot.PlotCrossSection(modelgrid=gwf.modelgrid, line={"line": [(spatial.x0, y),(spatial.x1, y)]},
                                        extent = [x0,x1,z0,z1], geographic_coords=True)


    csa = xsect.plot_array(a = a, cmap = structuralmodel.cmap, norm = structuralmodel.norm, 
                           alpha=0.8, vmin = vmin, vmax = vmax)
    
    ax.plot(x, z, 'o', color = 'red')
    ax.set_xlabel('x (m)', size = 10)
    ax.set_ylabel('z (m)', size = 10)
    linecollection = xsect.plot_grid(lw = 0.1, color = 'black') 
    
    