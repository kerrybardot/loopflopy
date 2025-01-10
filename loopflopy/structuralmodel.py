import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

class StructuralModel:
    def __init__(self, spatial, geodata_fname, data_sheetname, strat_sheetname):
        self.geodata_fname = geodata_fname
        self.data_sheetname = data_sheetname
        self.strat_sheetname = strat_sheetname
        self.origin = np.array([spatial.x0, spatial.y0, spatial.z0]).astype(float)
        self.maximum = np.array([spatial.x1, spatial.y1, spatial.z1]).astype(float)

        self.x0, self.y0, self.z0 = spatial.x0, spatial.y0, spatial.z0
        self.x1, self.y1, self.z1 = spatial.x1, spatial.y1, spatial.z1
        

    def make_cmap(self): 
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
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    def plot_transects(self, transect_x, transect_y, nx, ny, nz):
        
        # N-S TRANSECTS
        x0, y0, z0 = self.x0, self.y0, self.z0
        x1, y1, z1 = self.x1, self.y1, self.z1

        z = np.linspace(z0, z1, nz)
        y = np.linspace(y0, y1, ny)
        Y,Z = np.meshgrid(y,z)

        labels = self.strat_names[1:]
        ticks = [i + 0.5 for i in np.arange(0,len(labels))]
        labels = self.strat_names[1:]
        
        plt.figure(figsize=(10, 8))
        for i, n in enumerate(transect_x):
            X = np.zeros_like(Y)
            X[:,:] = n
            plt.subplot(len(transect_x), 1, i+1)
            print(X.flatten().shape,Y.flatten().shape,Z.flatten().shape)
            V = self.model.evaluate_model(np.array([X.flatten(),Y.flatten(),Z.flatten()]).T).reshape(np.shape(X))
            csa = plt.imshow(np.ma.masked_where(V<0,V), origin = "lower", extent = [y0,y1,z0,z1], aspect = 'auto', cmap = self.cmap, norm = self.norm)
            if i < (len(transect_x)-1):
                plt.xticks(ticks = [], labels = [])
            else:
                plt.xlabel('Easting (m)')
            #cbar = plt.colorbar(shrink = 0.9)
            #cbar.ax.set_yticks(ticks = ticks, labels = self.strat_names, size = 10) #verticalalignment = 'center')
            
            cbar = plt.colorbar(csa,
                                boundaries=np.arange(0,len(labels)+1),
                                shrink = 0.8
                                )
            cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 10, verticalalignment = 'center')    
            plt.title("x = " + str(transect_x[i]), size = 8)
            plt.ylabel('Elev. (mAHD)')
        plt.show()
        
        # W-E TRANSECTS
        
        z = np.linspace(z0, z1, nz)
        x = np.linspace(x0, x1, nx)
        X,Z = np.meshgrid(x,z)
        
        plt.figure(figsize=(10, 8))
        for i, n in enumerate(transect_y):
            Y = np.zeros_like(X)
            Y[:,:] = n
            plt.subplot(len(transect_y), 1, i+1)
            V = self.model.evaluate_model(np.array([X.flatten(),Y.flatten(),Z.flatten()]).T).reshape(np.shape(Y))
            plt.imshow(np.ma.masked_where(V<0,V), origin = "lower", extent = [x0,x1,z0,z1], cmap = self.cmap, norm = self.norm, aspect = 'auto') 
            if i < (len(transect_y)-1):
                plt.xticks(ticks = [], labels = [])
            else:
                plt.xlabel('Northing (m)')
            cbar = plt.colorbar(csa,
                                boundaries=np.arange(0,len(labels)+1),
                                shrink = 0.8
                                )
            cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 10, verticalalignment = 'center')    
            plt.title("y = " + str(transect_y[i]), size = 8)
            plt.ylabel('Elev. (mAHD)')
        plt.show()