import sys
import os
import flopy
import rasterio
import pickle
import matplotlib.pyplot as plt
import numpy as np

class DEM:
    
    def __init__(self, geotiff_fname):   
        self.geotiff_fname = geotiff_fname

    def plot_geotiff(self):
        with rasterio.open(self.geotiff_fname) as src:
            data = src.read(1)
            nodata_value = src.nodata
            print(nodata_value)
        if nodata_value is not None:
            masked_data = np.ma.masked_equal(data, nodata_value)
        else:
            masked_data = data  # If no NoData value is set, use the data as is
        
        plt.figure(figsize=(10, 8))
        plt.imshow(masked_data, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Data Values', shrink = 0.5)
        plt.title('GeoTIFF Visualization (NoData Values Excluded)')
        plt.xlabel('Column Number')
        plt.ylabel('Row Number')
        plt.show()
    
    def resample_topo(self, project, mesh, crop_polygon, fname):
        fine_topo = flopy.utils.Raster.load(self.geotiff_fname)
        topo_cropped = fine_topo.crop(crop_polygon)
        print(mesh.vgrid.crs)
        print(fine_topo.crs)
        topo = fine_topo.resample_to_grid(mesh.vgrid, band=fine_topo.bands[0], method="linear", extrapolate_edges=True,)
        pickle.dump(topo, open(os.path.join(fname),'wb'))

    def load_topo(self, project, fname):
        pickleoff = open(fname,'rb')
        self.topo = pickle.load(pickleoff)
        pickleoff.close()

    def plot_topo(self, mesh, levels):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()
        ax.set_title('Top Elevation (m)')
        pmv = flopy.plot.PlotMapView(modelgrid=mesh.vgrid)
        t = pmv.plot_array(self.topo)#, ec="0.75")
        cbar = plt.colorbar(t, shrink = 0.5)  
        cg = pmv.contour_array(self.topo, levels=levels, linewidths=0.8, colors="0.75")
