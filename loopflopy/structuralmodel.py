import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

class StructuralModel:
    def __init__(self, spatial, bbox, geodata_fname, data_sheetname, strat_sheetname):
        self.geodata_fname = geodata_fname
        self.data_sheetname = data_sheetname
        self.strat_sheetname = strat_sheetname
        self.origin = bbox[0] #np.array([spatial.x0, spatial.y0, spatial.z0]).astype(float)
        self.maximum = bbox[1] #np.array([spatial.x1, spatial.y1, spatial.z1]).astype(float)

        self.x0, self.y0, self.z0 = bbox[0][0], bbox[0][1], bbox[0][2]
        self.x1, self.y1, self.z1 = bbox[1][0], bbox[1][1], bbox[1][2]
        

    '''def make_cmap(self): 
        stratcolors = []
        for i in range(1,len(self.strat)):
            R = self.strat.R.loc[i].item() / 255
            G = self.strat.G.loc[i].item() / 255
            B = self.strat.B.loc[i].item() / 255
            stratcolors.append([round(R, 2), round(G, 2), round(B, 2)])
        nlg = len(self.strat_names[1:]) # number of layers geologic (Don't include above ground)
        cvals = np.arange(1,nlg) 
        norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), stratcolors))
        self.norm = norm
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)'''

    def plot_xtransects(self, transect_x, ny, nz, **kwargs):
        
        y0 = kwargs.get('y0', self.y0)
        z0 = kwargs.get('z0', self.z0)
        y1 = kwargs.get('y1', self.y1)
        z1 = kwargs.get('z1', self.z1)
            
        z = np.linspace(z0, z1, nz)
        y = np.linspace(y0, y1, ny)
        Y,Z = np.meshgrid(y,z)

        labels = self.strat_names[1:]
        ticks = [i for i in np.arange(0,len(labels))]
        boundaries = np.arange(-1,len(labels),1)+0.5
        
        plt.figure(figsize=(12, 8))
        for i, n in enumerate(transect_x):
            X = np.zeros_like(Y)
            X[:,:] = n
            plt.subplot(len(transect_x), 1, i+1)
            #print(X.flatten().shape,Y.flatten().shape,Z.flatten().shape)
            V = self.model.evaluate_model(np.array([X.flatten(),Y.flatten(),Z.flatten()]).T).reshape(np.shape(X))
            csa = plt.imshow(np.ma.masked_where(V<0,V), origin = "lower", extent = [y0,y1,z0,z1], aspect = 'auto', cmap = self.cmap, norm = self.norm)
            if i < (len(transect_x)-1):
                plt.xticks(ticks = [], labels = [])
            else:
                plt.xlabel('Easting (m)')
            
            cbar = plt.colorbar(csa,
                                boundaries = boundaries,
                                shrink = 1.0
                                )
            cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')    
            plt.title("x = " + str(transect_x[i]), size = 8)
            plt.ylabel('Elev. (mAHD)')
        plt.show()
        
    def plot_ytransects(self, transect_y, nx, nz, **kwargs):
        
        x0 = kwargs.get('x0', self.x0)
        z0 = kwargs.get('z0', self.z0)
        x1 = kwargs.get('x1', self.x1)
        z1 = kwargs.get('z1', self.z1)
        
        z = np.linspace(z0, z1, nz)
        x = np.linspace(x0, x1, nx)
        X,Z = np.meshgrid(x,z)

        labels = self.strat_names[1:]
        ticks = [i for i in np.arange(0,len(labels))]
        boundaries = np.arange(-1,len(labels),1)+0.5

        
        for i, n in enumerate(transect_y):
            fig = plt.figure(figsize=(12, 8))
            ax = plt.subplot(len(transect_y), 1, i+1)
            Y = np.zeros_like(X)
            Y[:,:] = n

            # Evaluate model to plot lithology
            V = self.model.evaluate_model(np.array([X.flatten(),Y.flatten(),Z.flatten()]).T).reshape(np.shape(Y))
            csa = ax.imshow(np.ma.masked_where(V<0,V), origin = "lower", extent = [x0,x1,z0,z1], cmap = self.cmap, norm = self.norm, aspect = 'auto') 

            # Evaluate faults to plot
            for fault in self.faults:
                F = self.model.evaluate_feature_value(fault, np.array([X.flatten(),Y.flatten(),Z.flatten()]).T).reshape(np.shape(Y))
                ax.contour(X, Z, F, levels = [0], colors = 'Black', linewidths=2., linestyles = 'dashed') 
            if i < (len(transect_y)-1):
                ax.set_xticks(ticks = [], labels = [])
            else:
                ax.set_xlabel('Northing (m)')
            cbar = plt.colorbar(csa,
                                ax=ax,
                                boundaries=boundaries,
                                shrink = 1.0
                                )
            cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')    
            ax.set_title("y = " + str(transect_y[i]), size = 8)
            ax.set_ylabel('Elev. (mAHD)')
            plt.show()