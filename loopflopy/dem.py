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
    """
    A digital elevation model (DEM) processing class for groundwater modeling.
    
    This class provides functionality for loading, processing, and resampling
    digital elevation data from GeoTIFF files for use in groundwater flow models.
    It handles coordinate system transformations, cropping, and integration with
    computational meshes.
    
    Parameters
    ----------
    geotiff_fname : str
        Path to the input GeoTIFF file containing elevation data.
    
    Attributes
    ----------
    geotiff_fname : str
        Path to the source GeoTIFF file.
    topo : ndarray
        Resampled topographic data array (set after calling load_topo()).
    
    Notes
    -----
    The class supports common DEM operations including:
    - Loading elevation data from GeoTIFF files
    - Cropping to specific geographic extents
    - Resampling to computational mesh grids
    - Visualization and quality control
    - Coordinate system handling
    
    Typical workflow:
    1. Initialize with GeoTIFF file path
    2. Optionally crop to study area using crop_raster()
    3. Resample to mesh resolution using resample_topo()
    4. Load resampled data using load_topo()
    5. Visualize results using plot_topo()
    
    Examples
    --------
    >>> # Basic DEM processing workflow
    >>> dem = DEM('elevation_data.tif')
    >>> 
    >>> # Crop to study area
    >>> dem.crop_raster('study_area.shp', 'cropped_elevation.tif')
    >>> 
    >>> # Resample to mesh and save
    >>> dem.resample_topo(mesh, 'resampled_topo.pkl')
    >>> 
    >>> # Load and visualize
    >>> dem.load_topo('resampled_topo.pkl')
    >>> levels = np.arange(0, 500, 25)  # 25m contours
    >>> dem.plot_topo(mesh, levels)
    
    See Also
    --------
    Mesh : For computational grid generation
    rasterio : For raster data I/O operations
    flopy.utils.Raster : For FloPy raster utilities
    """
    
    def __init__(self, geotiff_fname):   
        self.geotiff_fname = geotiff_fname

    def crop_raster(self, bbox_path, cropped_raster_path):
        """
        Crop the DEM to a specified geographic boundary.
        
        Clips the elevation raster to the extent defined by a shapefile polygon,
        reducing file size and focusing on the study area of interest.
        
        Parameters
        ----------
        bbox_path : str
            Path to shapefile containing the boundary polygon for cropping.
        cropped_raster_path : str
            Output path for the cropped GeoTIFF file.
        
        Notes
        -----
        The method:
        1. Reads boundary geometry from the input shapefile
        2. Uses rasterio.mask to clip the raster data
        3. Updates metadata for the cropped extent
        4. Saves the result as a new GeoTIFF file
        
        The cropped raster retains the same resolution and coordinate system
        as the original but covers only the specified geographic area.
        
        Examples
        --------
        >>> dem = DEM('regional_elevation.tif')
        >>> dem.crop_raster('model_boundary.shp', 'local_elevation.tif')
        """
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
        """
        Create a visualization of the GeoTIFF elevation data.
        
        Displays the elevation data as a colored image with proper handling
        of NoData values and a colorbar for reference.
        
        Notes
        -----
        The plot features:
        - Color-coded elevation values using 'viridis' colormap
        - Automatic masking of NoData values
        - Colorbar showing elevation scale
        - Row/column coordinate axes
        
        This method is useful for:
        - Initial data quality assessment
        - Checking for data gaps or artifacts
        - Understanding elevation distribution
        - Verifying correct data loading
        
        Examples
        --------
        >>> dem = DEM('elevation.tif')
        >>> dem.plot_geotiff()  # Display elevation visualization
        """
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
        """
        Resample elevation data to match the computational mesh resolution.
        
        Interpolates the DEM data onto the mesh cell centers, providing
        elevation values for each model cell. This is essential for setting
        up groundwater models with realistic topography.
        
        Parameters
        ----------
        mesh : Mesh
            Computational mesh object containing grid structure and coordinates.
        output_fname : str
            Path to save the resampled topography data (as pickle file).
        crop_polygon : shapely.geometry.Polygon or False, optional
            Polygon to crop the raster before resampling (default: False).
        method : str, optional
            Interpolation method for resampling (default: "nearest").
            Options: "nearest", "linear", "cubic".
        extrapolate_edges : bool, optional
            Whether to extrapolate values at mesh edges (default: True).
        
        Notes
        -----
        Resampling Methods:
        - "nearest": Fast, preserves original values, good for categorical data
        - "linear": Smoother results, good for continuous elevation data
        - "cubic": Smoothest results, slower computation
        
        The method:
        1. Loads the raster using FloPy utilities
        2. Optionally crops to specified polygon
        3. Resamples to mesh cell centers using specified interpolation
        4. Saves results as pickle file for later use
        
        Edge extrapolation helps ensure complete coverage when the mesh
        extends slightly beyond the DEM extent.
        
        Examples
        --------
        >>> # Basic resampling with nearest neighbor
        >>> dem.resample_topo(mesh, 'mesh_topo.pkl')
        >>>
        >>> # High-quality resampling with linear interpolation
        >>> dem.resample_topo(mesh, 'smooth_topo.pkl', method='linear')
        >>>
        >>> # Crop and resample to study area
        >>> from shapely.geometry import Polygon
        >>> study_area = Polygon([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
        >>> dem.resample_topo(mesh, 'cropped_topo.pkl', crop_polygon=study_area)
        """
        topo = flopy.utils.Raster.load(self.geotiff_fname)
        if crop_polygon:
            topo = topo.crop(crop_polygon)
        print(mesh.vgrid.crs)
        print(topo.crs)
        resampled_topo = topo.resample_to_grid(mesh.vgrid, band=topo.bands[0], method=method, extrapolate_edges=extrapolate_edges,)
        pickle.dump(resampled_topo, open(os.path.join(output_fname),'wb'))

    def load_topo(self, fname):
        """
        Load previously resampled topography data from pickle file.
        
        Loads elevation data that was resampled to mesh resolution and
        stored using the resample_topo() method.
        
        Parameters
        ----------
        fname : str
            Path to the pickle file containing resampled topography data.
        
        Sets Attributes
        ---------------
        topo : ndarray
            Array of elevation values for each mesh cell.
        
        Examples
        --------
        >>> dem = DEM('elevation.tif')
        >>> dem.load_topo('mesh_topo.pkl')
        >>> print(f"Elevation range: {dem.topo.min():.1f} to {dem.topo.max():.1f} m")
        """
        pickleoff = open(fname,'rb')
        self.topo = pickle.load(pickleoff)
        pickleoff.close()

    def plot_topo(self, mesh, levels):
        """
        Plot the resampled topography over the computational mesh.
        
        Creates a plan view visualization of elevation data with optional
        contour lines, showing how topography varies across the model domain.
        
        Parameters
        ----------
        mesh : Mesh
            Computational mesh object for spatial reference.
        levels : array_like
            Elevation contour levels to display (e.g., [0, 25, 50, 75, 100]).
        
        Notes
        -----
        The plot includes:
        - Color-filled elevation values across mesh cells
        - Contour lines at specified elevation levels (if not transect)
        - Colorbar showing elevation scale
        - Proper spatial coordinate system
        
        Contours are automatically styled with:
        - Blue color scheme ('b')
        - Terrain colormap for filled contours
        - 0.8 pt line width
        
        This visualization is essential for:
        - Verifying correct topography assignment
        - Understanding terrain influence on groundwater flow
        - Quality control of resampling process
        - Presentation of model setup
        
        Examples
        --------
        >>> # Plot with 25m contour intervals
        >>> levels = np.arange(0, 500, 25)
        >>> dem.plot_topo(mesh, levels)
        >>>
        >>> # Plot with custom contour levels
        >>> levels = [10, 50, 100, 200, 300]
        >>> dem.plot_topo(mesh, levels)
        """
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()
        ax.set_title('Top Elevation (m)')
        pmv = flopy.plot.PlotMapView(modelgrid=mesh.vgrid)
        t = pmv.plot_array(self.topo)#, ec="0.75")
        cbar = plt.colorbar(t, shrink = 0.5)  
        if mesh.plangrid != 'transect':
            cg = pmv.contour_array(self.topo, levels=levels, cmap = 'terrain', linewidths=0.8, colors='b')#"0.75")

def crop_geotiff(original_tif_path, crop_polygon_shp_path, output_tif_path, crs = 28350):
    """
    Crop a GeoTIFF file using a polygon shapefile boundary.
    
    Standalone function to clip raster data to a specified geographic area
    defined by a shapefile polygon. Ensures consistent coordinate reference
    systems and proper NoData handling.
    
    Parameters
    ----------
    original_tif_path : str
        Path to the input GeoTIFF file to be cropped.
    crop_polygon_shp_path : str
        Path to shapefile containing the boundary polygon.
    output_tif_path : str
        Path for the output cropped GeoTIFF file.
    crs : int, optional
        EPSG code for coordinate reference system (default: 28350).
    
    Notes
    -----
    The function:
    1. Reads the boundary shapefile and reprojects to specified CRS
    2. Clips the raster using the polygon boundary
    3. Sets NoData value to -9999 for consistency
    4. Saves the cropped result with updated metadata
    
    This is useful for preprocessing large regional datasets to focus
    on specific study areas, reducing file sizes and processing time.
    
    Examples
    --------
    >>> # Crop regional DEM to local study area
    >>> crop_geotiff('regional_dem.tif', 'study_boundary.shp', 
    ...              'local_dem.tif', crs=32750)
    """

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
    Display comprehensive coordinate reference system information for a GeoTIFF.
    
    Analyzes and reports detailed CRS information including EPSG codes,
    projection types, spatial extent, and resolution for quality control
    and compatibility checking.
    
    Parameters
    ----------
    file_path : str
        Path to the GeoTIFF file to analyze.
    
    Notes
    -----
    Reports include:
    - CRS definition and EPSG code
    - Geographic vs projected coordinate system type
    - Raster dimensions (width x height)
    - Spatial bounds (extent)
    - Pixel resolution
    
    This function is essential for:
    - Verifying coordinate systems before processing
    - Checking compatibility with other spatial data
    - Understanding data characteristics
    - Troubleshooting projection issues
    
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
    Convert a GeoTIFF from one coordinate reference system to another.
    
    Reprojects raster data to a different CRS while maintaining data integrity
    and allowing selection of appropriate resampling methods for the data type.
    
    Parameters
    ----------
    input_path : str
        Path to the input GeoTIFF file.
    output_path : str
        Path for the output reprojected GeoTIFF file.
    target_crs : str, int, or pyproj.CRS
        Target coordinate reference system. Can be:
        - EPSG code as integer (e.g., 4326, 32750)
        - EPSG string (e.g., 'EPSG:32750')
        - Proj4 string
        - pyproj CRS object
    resampling_method : str, optional
        Resampling algorithm for reprojection (default: 'nearest').
        Options: 'nearest', 'bilinear', 'cubic', 'average', 'mode', 'lanczos'.
    
    Returns
    -------
    bool
        True if conversion successful, False if error occurred.
    
    Notes
    -----
    Resampling Method Selection:
    - 'nearest': Best for categorical data, preserves original values
    - 'bilinear': Good balance for continuous data like elevation
    - 'cubic': Smoothest results, best for high-quality elevation data
    - 'average': Good for downsampling, reduces noise
    - 'mode': Best for categorical data when upsampling
    - 'lanczos': High-quality for photographic imagery
    
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
    Analyze and display comprehensive CRS information for a shapefile.
    
    Provides detailed coordinate reference system information, geometry
    statistics, and spatial extent for vector data quality control and
    compatibility verification.
    
    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile (.shp) to analyze.
    
    Returns
    -------
    geopandas.GeoDataFrame or None
        The loaded GeoDataFrame if successful, None if error occurred.
    
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
    Assign or update the coordinate reference system of a GeoTIFF file.
    
    Sets the CRS metadata for a raster file without reprojecting the data.
    This is useful when the CRS information is missing or incorrect but
    the coordinate values are already in the desired system.
    
    Parameters
    ----------
    input_path : str
        Path to the input GeoTIFF file.
    output_path : str
        Path for the output GeoTIFF with updated CRS.
    target_crs : str or int
        Target coordinate reference system specification.
        Can be EPSG code, Proj4 string, or WKT string.
    
    Notes
    -----
    **Important:** This function only updates the CRS metadata without
    transforming coordinate values. Use this when:
    - The raster has correct coordinates but missing/wrong CRS info
    - You know the true coordinate system of the data
    - The spatial data aligns correctly with reference data
    
    **Do not use this for actual coordinate transformation** - use
    convert_geotiff_crs() instead for reprojection.
    
    Examples
    --------
    >>> # Set CRS for raster with missing projection info
    >>> set_geotiff_crs('no_crs_dem.tif', 'utm_dem.tif', 'EPSG:32750')
    >>>
    >>> # Correct wrong CRS metadata
    >>> set_geotiff_crs('wrong_crs.tif', 'correct_crs.tif', 4326)
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