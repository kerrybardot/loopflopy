
class Spatial:
    """
    A spatial data management class for groundwater modeling projects.
    
    This class serves as a container for spatial data, coordinate systems, and
    geometric features used in groundwater flow modeling with LoopFlopy. It manages
    coordinate reference systems and provides a foundation for storing spatial
    datasets like boundaries, wells, faults, and other geographic features.
    
    Parameters
    ----------
    epsg : int
        EPSG code for the coordinate reference system (CRS) to be used for
        the modeling project. Eg:
        - 32750: WGS 84 / UTM zone 50S (Perth, Australia)
    
    Attributes
    ----------
    epsg : int
        EPSG code for the coordinate reference system.
    crs : int
        Alias for epsg, used for compatibility with GeoPandas and other libraries.
    
    Notes
    -----
    This class is designed to be extended with additional spatial data attributes
    such as:
    
    Typical Spatial Data Attributes (added dynamically):
    - Model boundaries: model_boundary_poly, inner_boundary_poly
    - Well locations: xyobsbores, xypumpbores, pumpbore_gdf, obsbore_gdf
    - Geological features: faults_gdf, river_gdf, geobore_gdf
    - Boundary conditions: chd_*_ls, ghb_*_ls for boundary line segments
    - Domain extents: x0, x1, y0, y1 for model bounds
    - Special polygons: Various *_poly attributes for constraint areas
    
    The class provides a centralized way to manage all spatial data and ensure
    consistent coordinate reference systems across the modeling workflow.
    
    Examples
    --------
    >>> # Create spatial container for Perth, Australia project
    >>> spatial = Spatial(epsg=32750)  # UTM Zone 50S
    >>> print(f"Using CRS: EPSG:{spatial.epsg}")
    
    >>> # Typically extended with spatial data:
    >>> spatial.x0, spatial.x1 = 700000, 710000  # Model extent in UTM
    >>> spatial.y0, spatial.y1 = 6200000, 6210000
    >>> spatial.xyobsbores = [(705000, 6205000), (708000, 6207000)]  # Well coords
    
    See Also
    --------
    Mesh : Uses spatial data for mesh generation and boundary identification
    Geomodel : Uses spatial boundaries for geological model evaluation
    geopandas.GeoDataFrame : For storing spatial data with proper CRS handling
    """
    
    def __init__(self, epsg):
        """
        Initialize Spatial object with coordinate reference system.
        
        Parameters
        ----------
        epsg : int
            EPSG code for the coordinate reference system.
        
        Notes
        -----
        Sets both epsg and crs attributes to the same value for compatibility
        with different spatial libraries and conventions.
        """       
        self.epsg = epsg
        self.crs = epsg

# -----------------------------------------------



