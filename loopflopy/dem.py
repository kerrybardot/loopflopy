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
from rasterio.warp import calculate_default_transform, reproject, Resampling


class DEM:
    
    def __init__(self, geotiff_fname):   
        self.geotiff_fname = geotiff_fname

    def crop_raster(self, bbox_path, cropped_raster_path):
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
        
        with rasterio.open(cropped_raster_path, "w", **out_meta) as dest:
            print(dest.crs)
            dest.write(out_image)

    def plot_geotiff(self):
        with rasterio.open(self.geotiff_fname) as src:
            data = src.read(1)
            nodata_value = src.nodata
            #print(nodata_value)
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
    
    def resample_topo(self, mesh, output_fname, crop_polygon = False, method="nearest", extrapolate_edges=True,):
        topo = flopy.utils.Raster.load(self.geotiff_fname)
        if crop_polygon:
            topo = topo.crop(crop_polygon)
        print(mesh.vgrid.crs)
        print(topo.crs)
        resampled_topo = topo.resample_to_grid(mesh.vgrid, band=topo.bands[0], method=method, extrapolate_edges=extrapolate_edges,)
        pickle.dump(resampled_topo, open(os.path.join(output_fname),'wb'))

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
        if mesh.plangrid != 'transect':
            cg = pmv.contour_array(self.topo, levels=levels, cmap = 'terrain', linewidths=0.8, colors='b')#"0.75")

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

def check_geotiff_crs(file_path):
    """
    Check the CRS of a GeoTIFF file.
    
    Parameters:
    -----------
    file_path : str
        Path to GeoTIFF file
    """
    try:
        with rasterio.open(file_path) as src:
            print(f"\n📍 CRS Information for: {file_path}")
            print(f"CRS: {src.crs}")
            print(f"EPSG code: {src.crs.to_epsg() if src.crs else 'No EPSG code'}")
            print(f"Is geographic: {src.crs.is_geographic if src.crs else 'Unknown'}")
            print(f"Is projected: {src.crs.is_projected if src.crs else 'Unknown'}")
            print(f"Shape: {src.width} x {src.height}")
            print(f"Bounds: {src.bounds}")
            print(f"Resolution: {src.res}")
            
    except Exception as e:
        print(f"Error reading GeoTIFF: {str(e)}")

def convert_geotiff_crs(input_path, output_path, target_crs, resampling_method='nearest'):
    """
    Convert a GeoTIFF from one CRS to another.
    
    Parameters:
    -----------
    input_path : str
        Path to input GeoTIFF file
    output_path : str
        Path for output GeoTIFF file
    target_crs : str or int
        Target CRS (e.g., 'EPSG:32750', 4326, or pyproj CRS object)
    resampling_method : str
        Resampling method ('nearest', 'bilinear', 'cubic', 'average', etc.)
    
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Map resampling method names to rasterio constants
        resampling_methods = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'average': Resampling.average,
            'mode': Resampling.mode,
            'lanczos': Resampling.lanczos
        }
        
        resampling = resampling_methods.get(resampling_method.lower(), Resampling.nearest)
        
        with rasterio.open(input_path) as src:
            # Print original CRS info
            print(f"Original CRS: {src.crs}")
            print(f"Original shape: {src.width} x {src.height}")
            print(f"Original bounds: {src.bounds}")
            
            # Calculate the transform and dimensions for the new CRS
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            
            # Copy metadata and update for new CRS
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            print(f"Target CRS: {target_crs}")
            print(f"Target shape: {width} x {height}")
            
            # Create output file and reproject
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=resampling
                    )
            
            print(f"Successfully converted {input_path} to {output_path}")
            return True
            
    except Exception as e:
        print(f"Error converting GeoTIFF: {str(e)}")
        return False
    
def check_shapefile_crs(shapefile_path):
    """
    Check the CRS of a shapefile and display detailed information
    """
    try:
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        print(f"Shapefile: {shapefile_path}")
        print("-" * 50)
        
        # Basic CRS information
        print(f"CRS: {gdf.crs}")
        print(f"CRS name: {gdf.crs.name if gdf.crs else 'No CRS defined'}")
        print(f"EPSG code: {gdf.crs.to_epsg() if gdf.crs else 'No EPSG code'}")
        
        # Detailed CRS information
        if gdf.crs:
            print(f"CRS as string: {gdf.crs.to_string()}")
            print(f"CRS as proj4: {gdf.crs.to_proj4()}")
            print(f"Is geographic (lat/lon): {gdf.crs.is_geographic}")
            print(f"Is projected: {gdf.crs.is_projected}")
            print(f"Units: {gdf.crs.axis_info[0].unit_name if gdf.crs.axis_info else 'Unknown'}")
        else:
            print("⚠️  No CRS defined for this shapefile!")
            
        # Geometry information
        print(f"\nGeometry Info:")
        print(f"Number of features: {len(gdf)}")
        print(f"Geometry types: {gdf.geometry.type.unique()}")
        print(f"Bounds: {gdf.total_bounds}")
        
        return gdf
        
    except Exception as e:
        print(f"❌ Error reading shapefile: {e}")
        return None

def set_geotiff_crs(input_path, output_path, target_crs):
    """
    Set/update the CRS of a GeoTIFF file
    """
    with rasterio.open(input_path) as src:
        # Read all the raster data and metadata
        data = src.read()
        profile = src.profile.copy()
        
        # Update the CRS
        profile['crs'] = CRS.from_string(target_crs)
        
        # Write to new file with updated CRS
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
    
    print(f"✅ CRS updated to {target_crs}")
    print(f"Output saved: {output_path}")