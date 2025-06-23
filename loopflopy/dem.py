import sys
import os
import flopy
import rasterio
import pickle
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import mapping
import geopandas as gpd
from rasterio.mask import mask
from pyproj import CRS
import fiona


class DEM:
    
    def __init__(self, geotiff_fname):   
        self.geotiff_fname = geotiff_fname

    def crop_raster(self, bbox_path):
        with fiona.open(bbox_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        
        with rasterio.open(self.geotiff_fname) as src:
            print(src.crs)
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta
        
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
        
        with rasterio.open("../data/dem/cropped_raster.tif", "w", **out_meta) as dest:
            print(dest.crs)
            dest.write(out_image)

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
    
    def resample_topo(self, mesh, fname, crop_polygon = False, method="nearest", extrapolate_edges=True,):
        topo = flopy.utils.Raster.load(self.geotiff_fname)
        if crop_polygon:
            topo = topo.crop(crop_polygon)
        print(mesh.vgrid.crs)
        print(topo.crs)
        resampled_topo = topo.resample_to_grid(mesh.vgrid, band=topo.bands[0], method=method, extrapolate_edges=extrapolate_edges,)
        pickle.dump(resampled_topo, open(os.path.join(fname),'wb'))

    def load_topo(self, fname):
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
        cg = pmv.contour_array(self.topo, levels=levels, linewidths=0.8, colors='b')#"0.75")

def crop_geotiff(original_tif_path, crop_polygon_shp_path, output_tif_path, crs = 28350):

    shapefile = gpd.read_file(crop_polygon_shp_path)
    shapefile = shapefile.to_crs(CRS.from_epsg(crs)) #Keep all CRS the same

    with rasterio.open(original_tif_path) as src:
        
        shapes = [mapping(geom) for geom in shapefile.geometry] # Convert shapefile geometry to GeoJSON-like mapping
        out_image, out_transform = mask(src, shapes, crop=True) # Clip the raster with the shapefile
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": -9999
        })

    # Save the clipped raster
    with rasterio.open(output_tif_path, "w", **out_meta) as dest:
        dest.write(out_image)

    nodata_val = out_meta.get("nodata", -9999)
    out_image = np.ma.masked_equal(out_image, nodata_val)
    
    print(f"Clipped GeoTIFF saved to: {output_tif_path}")