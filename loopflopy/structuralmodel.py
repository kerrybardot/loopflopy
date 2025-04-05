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

    def plot_xytransect(self, start, end, z0, z1, nh, nz, **kwargs):
    
        x0 = start[0]
        y0 = start[1]
        x1 = end[0]
        y1 = end[1]

        x = np.linspace(x0, x1, nh)
        y = np.linspace(y0, y1, nh)        
        z = np.linspace(z0, z1, nz)

        X = np.tile(x, (len(z), 1)) 
        Y = np.tile(y, (len(z), 1)) 
        Z = np.tile(z[:, np.newaxis], (1, nh))  # Repeat z along columns (nh times)

        labels = self.strat_names[1:]
        ticks = [i for i in np.arange(0,len(labels))]
        boundaries = np.arange(-1,len(labels),1)+0.5

        a = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
        V = self.model.evaluate_model(a).reshape(np.shape(X))

        plt.figure(figsize=(8, 5))
        plt.subplot(111)

        csa = plt.imshow(np.ma.masked_where(V<0,V), origin = "lower", extent = [y0,y1,z0,z1], aspect = 'auto', cmap = self.cmap, norm = self.norm)
        
        cbar = plt.colorbar(csa,
                            boundaries = boundaries,
                            shrink = 1.0
                            )
        cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')    
        #plt.xticks(ticks = [], labels = [])
        plt.xlabel('Easting (m)')
        plt.title("Transect", size = 8)
        plt.ylabel('Elev. (mAHD)')
        plt.savefig('../figures/structural_xytransect.png')
        plt.show()
    
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
        
        plt.figure(figsize=(8, 5))
        for i, n in enumerate(transect_x):
            X = np.zeros_like(Y)
            X[:,:] = n
            plt.subplot(len(transect_x), 1, i+1)
            a = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
            print(type(a))
        
            print(type(a[0]))
            print(X.flatten().shape,Y.flatten().shape,Z.flatten().shape)
           
            V = self.model.evaluate_model(np.array([X.flatten(),Y.flatten(),Z.flatten()]).T).reshape(np.shape(X))
            print('V shape', V.shape)
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
        plt.savefig('../figures/structural_xtransects.png')
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
            fig = plt.figure(figsize=(8, 5))
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
            plt.savefig('../figures/structural_ytransects.png')
            plt.show()
        
    def plot_ytransects2(self, transect_y, nx, nz, **kwargs):

        self.sequence_names = []
        for item in self.strat['sequence'].tolist():
            if item not in self.sequence_names:
                self.sequence_names.append(item)
        #print(self.sequence_names)

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
            fig = plt.figure(figsize=(12, 3))
            ax = plt.subplot(len(transect_y), 1, i+1)
            ax.set_aspect('equal')
            Y = np.zeros_like(X)
            Y[:,:] = n

            # Evaluate model to plot lithology
            V = self.model.evaluate_model(np.array([X.flatten(),Y.flatten(),Z.flatten()]).T).reshape(np.shape(Y))
            csa = ax.imshow(np.ma.masked_where(V<0,V), origin = "lower", extent = [x0,x1,z0,z1], cmap = self.cmap, norm = self.norm, aspect = 'auto') 
            #for val in self.strat['val'].tolist():
            #    ax.contour(X, Z, V, levels = [val], colors = 'Black', linewidths=1., linestyles = 'dashed') 

            val_above = 0
            # Evaluate scalar fields for each feature to plot contours
            for k, feat in enumerate(self.sequence_names[1:]):
                
                values = self.strat[self.strat['sequence'] == feat].val.tolist()
                values = sorted(values)
                #print(feat, values, val_above)
                V = self.model.evaluate_feature_value(feat, np.array([X.flatten(),Y.flatten(),Z.flatten()]).T).reshape(np.shape(Y))
                for j in range(len(values)-1):
                    contour_values = np.arange(values[j], values[j+1], 20)
                    for val in contour_values:
                        #print(feat, val)
                        ax.contour(X, Z, V, levels = [val], colors = 'Black', linewidths=0.5, linestyles = 'dashed') 
                
                # plot contours fror top unit within feature
                contour_values = np.arange(values[-1], 9999, 20)#val_above+20, 20)
                for val in contour_values:
                    #print(feat, val)
                    ax.contour(X, Z, V, levels = [val], colors = 'Black', linewidths=0.5, linestyles = 'dashed') 
                val_above = values[0]
                
                

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
                                shrink = 0.5
                                )
            cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')    
            ax.set_title("y = " + str(transect_y[i]), size = 8)
            ax.set_ylabel('Elev. (mAHD)')
            #plt.axis('equal')
            plt.savefig('../figures/structural_ytransects.png')
            plt.show()