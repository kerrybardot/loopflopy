#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

def plot_bores(P):
    plt.figure(figsize=(3,3))
    
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    i = 0
    for (xi, yi) in zip(P.data.X, P.data.Y):
        if P.data.data_type[i] == 'raw':
            plt.scatter(P.data.X[i], P.data.Y[i], color = 'blue')#, size = 2)
            plt.text(xi, yi, P.data.ID[i], size = 11, va='bottom', ha='center')
        if P.data.data_type[i] == 'control':
            plt.scatter(P.data.X[i], P.data.Y[i], color = 'red')#, size = 2)
            plt.text(xi, yi, P.data.ID[i], size = 11, va='bottom', ha='center')
        if P.data.data_type[i] == 'fault_surface':
            plt.scatter(P.data.X[i], P.data.Y[i], color = 'purple')#, size = 1)
        i += 1
    plt.xlim(P.x0,P.x1)
    plt.ylim(P.y0,P.y1)
        

def plot_geo_2D(geomodel, X, upper_levels, lower_levels):
    #import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1,figsize=(5,2))
    r = 200
    # X section
    y = np.linspace(P.y0,P.y1,r)
    z = np.linspace(P.z0,P.z1,r)
    yy, zz = np.meshgrid(y,z)
    xx = np.zeros_like(yy)
    xx[:] = X
    ax.set_title('X = ' + str(X))

    vals = geomodel.evaluate_feature_value('upper',np.array([xx.flatten(),yy.flatten(),zz.flatten()]).T)
    ax.contourf(vals.reshape((r,r)), levels = upper_levels, extent=(P.y0, P.y1, P.z0, P.z1), cmap = 'Spectral')
    vals = geomodel.evaluate_feature_value('lower',np.array([xx.flatten(),yy.flatten(),zz.flatten()]).T)
    ax.contourf(vals.reshape((r,r)), levels = lower_levels, extent=(P.y0, P.y1, P.z0, P.z1))

    plt.show()
    
def plot_geo_2D_WE(geomodel, Y, upper_levels, lower_levels):
    #import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1,figsize=(5,2))
    r = 200
    #Y section
    x = np.linspace(P.x0,P.x1,r)
    z = np.linspace(P.z0,P.z1,r)
    xx, zz = np.meshgrid(x,z)
    yy = np.zeros_like(xx)
    yy[:] = Y
    ax.set_title('Y = ' + str(Y))

    vals = geomodel.evaluate_feature_value('upper',np.array([xx.flatten(),yy.flatten(),zz.flatten()]).T)
    ax.contourf(vals.reshape((r,r)), levels = upper_levels, extent=(P.x0, P.x1, P.z0, P.z1), cmap = 'Spectral')
    vals = geomodel.evaluate_feature_value('lower',np.array([xx.flatten(),yy.flatten(),zz.flatten()]).T)
    ax.contourf(vals.reshape((r,r)), levels = lower_levels, extent=(P.x0, P.x1, P.z0, P.z1))

    plt.show()
    
def create_faultfunction():    
    ## ADD FAULT (this chunk given to me directly by Lachlan Grose to make an ellipsoid fault)
    from LoopStructural.modelling.features.fault._fault_function import CubicFunction, FaultDisplacement, Composite
    hw = CubicFunction()
    hw.add_cstr(0, 1)
    hw.add_grad(0, 0)
    hw.add_cstr(1, 0)
    hw.add_grad(1, 0)
    hw.add_max(1)
    fw = CubicFunction()
    fw.add_cstr(0, -1)
    fw.add_grad(0, 0)
    fw.add_cstr(-1, 0)
    fw.add_grad(-1, 0)
    fw.add_min(-1)
    gyf = CubicFunction()
    gyf.add_cstr(-1, 0)
    gyf.add_cstr(1, 0)
    gyf.add_cstr(-0.2, 1)
    gyf.add_cstr(0.2, 1)
    gyf.add_grad(0, 0)
    gyf.add_min(-1)
    gyf.add_max(1)
    gzf = CubicFunction()
    gzf.add_cstr(-1, 0)
    gzf.add_cstr(1, 0)
    gzf.add_cstr(-0.2, 1)
    gzf.add_cstr(0.2, 1)
    gzf.add_grad(0, 0)
    gzf.add_min(-1)
    gzf.add_max(1)
    gxf = Composite(hw, fw)
    fault_displacement = None
    fault_displacement = FaultDisplacement(gx=gxf, gy=gyf, gz=gzf)
    faultfunction = fault_displacement
    return(faultfunction)


# In[ ]:


print('Geomodel routines loaded!')

