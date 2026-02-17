import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import flopy
import geopandas as gpd
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,MultiPoint,shape
from flopy.discretization import VertexGrid
from flopy.utils.triangle import Triangle as Triangle
from flopy.utils.voronoi import VoronoiGrid
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap

class Mesh:
    """
    A computational mesh class for creating 2D and 3D grids for MODFLOW 6 flows.
    
    This class handles creation of structured (Cartesian), triangular, and Voronoi meshes
    for use with MODFLOW 6. It supports both areal models and 2D transect models with
    various refinement options around wells and other features.
    
    Parameters
    ----------
    plangrid : str
        Type of horizontal discretization scheme:
        - 'car': Cartesian (structured) grid
        - 'tri': Triangular (unstructured) grid  
        - 'vor': Voronoi (unstructured) grid
        - 'transect': transect (structured) grid
    **kwargs
        Additional keyword arguments including:
        - special_cells: dict defining cells requiring special treatment
        - Grid-specific parameters (ncol, nrow, angle, radius1, radius2, etc.)
    
    Attributes
    ----------
    plangrid : str
        Grid type identifier (car, tri, vor, transect)
    ncpl : int
        Number of cells per layer in the horizontal plane.
    cell2d : list
        Cell connectivity information for DISV format.
    vertices : list
        Vertex coordinates for DISV format.
    xcyc : list
        Cell center coordinates as (x, y) tuples.
    xc, yc : list
        Separate lists of x and y cell center coordinates.
    vgrid : flopy.discretization.VertexGrid
        FloPy vertex grid object for visualization and intersection operations.
    gi : flopy.utils.GridIntersect
        Grid intersection utility for finding cells intersecting geometries.
    idomain : ndarray
        Active cell array (1=active, 0=inactive). Still need to upgrade car to exclude inactive cells.
    ibd : ndarray
        Special cell identification array for boundary conditions.
    
    Grid-Specific Attributes
    ------------------------
    For Cartesian grids:
        delx, dely : float
            Cell spacing in x and y directions.
        ncol, nrow : int
            Number of columns and rows.
        sg : flopy.discretization.StructuredGrid
            FloPy structured grid object.
    
    For Triangular/Voronoi grids:
        tri : flopy.utils.triangle.Triangle
            Triangle mesh generator object.
        vor : flopy.utils.voronoi.VoronoiGrid
            Voronoi grid generator object (for 'vor' type).
        nodes : ndarray
            Constraint nodes for mesh generation.
        polygons : list
            Constraint polygons for mesh generation.
    
    For Transect grids:
        length : float
            Total length of transect.
        angrot : float
            Rotation angle of transect from east.
        L : list
            Distance along transect for each cell center.
    
    Examples
    --------
    >>> # Create a triangular mesh
    >>> mesh = Mesh('tri', angle=30, modelmaxtri=5000)
    >>> mesh.prepare_nodes_and_polygons(spatial, 
    ...                                 ['boundary_nodes'], 
    ...                                 ['model_boundary_poly'])
    >>> mesh.create_mesh(project, spatial)
    >>>
    >>> # Create a transect mesh
    >>> mesh = Mesh('car')
    >>> mesh.create_mesh_transect(crs=32750, x0=0, x1=1000, y0=0, y1=0, 
    ...                          ncol=20, delc=50)
    >>>
    >>> # Locate special cells for boundary conditions
    >>> mesh.locate_special_cells(spatial, threshold=0.8)
    """    
    def __init__(self, plangrid, **kwargs):       
        self.plangrid = plangrid
        
        # Unpack kwargs
        special_cells = kwargs.get('special_cells', None)
        if special_cells is not None:
            self.special_cells = special_cells
        else:
            print("No special cells")
        
        #setattr(self, group, [])

#### 
    def create_bore_refinement(self, spatial):
        """
        Create mesh refinement nodes around pumping wells.
        
        Generates additional constraint nodes in concentric patterns around
        pumping well locations to ensure adequate mesh refinement for accurate
        simulation of well hydraulics.
        
        Parameters
        ----------
        spatial : Spatial
            Spatial data object containing pumping well coordinates.
        
        Notes
        -----
        For triangular grids ('tri'):
        - Creates triangular vertex patterns around each well
        - Uses two rings of triangular vertices at different radii
        
        For Voronoi grids ('vor'):
        - Creates circular patterns of nodes around each well
        - Uses single ring of evenly spaced nodes
        
        For Cartesian grids ('car'):
        - No refinement applied (structured grids have fixed geometry)
        
        The refinement radii are controlled by self.radius1 and self.radius2
        attributes which should be set during mesh initialization.
        
        Sets Attributes
        ---------------
        welnodes : list
            List of vertex coordinates for well refinement.
        spatial.bore_refinement_nodes : list
            Flattened list of all refinement nodes added to spatial object.
        """

        self.welnodes = []
        self.welnodes2 = []
        spatial.bore_refinement_nodes = []
        
        if not hasattr(spatial, 'xypumpbores'):
            print("No pumping bores defined in spatial object")
            return
        
        else:
            
            if self.plangrid == 'car':
                print('No bore refinement nodes for structured grids')
                
            if self.plangrid == 'tri':

                print("Creating bore refinement nodes for pumping bores")
                
                def verts1(X, Y, l): # l is distance from centre of triangle to vertex
                    x1 = X - l*3**0.5/2 
                    x2 = X + l*3**0.5/2 
                    x3 = X
                    y1 = Y - l/2
                    y2 = Y - l/2
                    y3 = Y + l
                    return(x1, x2, x3, y1, y2, y3)
                
                def verts2(X, Y, l): # l is distance from centre of triangle to vertex
                    x1 = X 
                    x2 = X + l*3**0.5
                    x3 = X - l*3**0.5
                    y1 = Y - 2*l
                    y2 = Y + l
                    y3 = Y + l
                    return(x1, x2, x3, y1, y2, y3)
        
                for i in spatial.xypumpbores:   
                    X, Y = i[0], i[1] # coord of pumping bore
                                
                    x1, x2, x3, y1, y2, y3 = verts1(X, Y, self.radius1) #/2
                    vertices1 = ((x1, y1), (x2, y2), (x3, y3))
                    x1, x2, x3, y1, y2, y3 = verts2(X, Y, self.radius1) #/2
                    vertices2 = ((x1, y1), (x2, y2), (x3, y3))
                    
                    self.welnodes.append(vertices1)
                    self.welnodes.append(vertices2)
                    
                for bore in range(2*spatial.npump): # x 2 because inner and outer ring of vertices
                    for node in range(len(self.welnodes[bore])):
                        spatial.bore_refinement_nodes.append(self.welnodes[bore][node])
                        
            if self.plangrid == 'vor':

                print("Creating bore refinement nodes for pumping bores")

                theta = np.linspace(0, 2 * np.pi, 11)
                for i in spatial.xypumpbores:   
                    X = i[0] + self.radius1 * np.cos(theta)
                    Y = i[1] + self.radius1 * np.sin(theta)    
                    vertices1 = [(x_val, y_val) for x_val, y_val in zip(X, Y)]
                    X = i[0] + self.radius2 * np.cos(theta)
                    Y = i[1] + self.radius2 * np.sin(theta)    
                    vertices2 = [(x_val, y_val) for x_val, y_val in zip(X, Y)]
                    self.welnodes.append(vertices1)
                    #self.welnodes2.append(vertices2)

                for bore in range(spatial.npump): # x 2 because inner and outer ring of vertices
                    for node in range(len(self.welnodes[bore])):
                        spatial.bore_refinement_nodes.append(self.welnodes[bore][node])

    def prepare_nodes_and_polygons(self, spatial, node_list, polygon_list):
        """
        Prepare constraint nodes and polygons for unstructured mesh generation.
        
        Collects constraint nodes and polygons from the spatial object to guide
        mesh generation for triangular and Voronoi grids. These constraints ensure
        mesh boundaries align with important features like faults, rivers, and
        model boundaries.
        
        Parameters
        ----------
        spatial : Spatial
            Spatial data object containing geometric features.
        node_list : list of str
            Names of spatial attributes containing constraint nodes.
            Each attribute should be a list of (x, y) coordinate tuples.
            Must end in '_nodes' to be recognized as node lists.
        polygon_list : list of str
            Names of spatial attributes containing constraint polygons.
            Each attribute should be a Shapely Polygon object.
            Must end in '_poly' to be recognized as polygon lists.
        
        Examples
        --------
        >>> mesh.prepare_nodes_and_polygons(spatial,
        ...     ['fault_nodes', 'river_nodes'],
        ...     ['model_boundary_poly', 'inner_boundary_poly'])
        
        Notes
        -----
        The method validates that:
        - Node attributes are lists of coordinate tuples
        - Polygon attributes are Shapely Polygon objects
        
        For polygons, it extracts:
        - Boundary coordinates from exterior ring
        - Representative point for area constraints
        - Maximum triangle area from self.modelmaxtri
        
        Sets Attributes
        ---------------
        nodes : ndarray
            Array of all constraint node coordinates.
        polygons : list
            List of polygon constraint tuples: (coords, point, max_area).
        """
        self.nodes = []
        for n in node_list: # e.g. n could be "faults_nodes"
            print(n)
            points = getattr(spatial, n)
            if type(points) == list:            
                for point in points: 
                    self.nodes.append(point)
            else:
                print("node_list ", n, " needs to be a list of tuples")
        self.nodes = np.array(self.nodes)

        self.polygons = [] # POLYGONS[(polygon, (x,y), maxtri)]
        for p in polygon_list: # e.g. p could be "model_boundary_poly"
            polygon = getattr(spatial, p)
            if type(polygon) == Polygon:            
                self.polygons.append((list(polygon.exterior.coords), 
                                      (polygon.representative_point().x, polygon.representative_point().y), 
                                       self.modelmaxtri)) 
            else:
                print("polygon ", n, " needs to be a Shapely Polygon")
            
    
    def create_mesh(self, project, spatial):
        """
        Create the computational mesh based on the specified grid type.
        
        Generates the appropriate mesh type (Cartesian, triangular, or Voronoi)
        using the prepared constraints and spatial boundaries. Creates all necessary
        grid data structures for use with FloPy and MODFLOW 6.

        If Transect grid is specified, use create_mesh_transect() or
        create_irregular_mesh_transect() instead.
        
        Parameters
        ----------
        project : Project
            Project object containing workspace and executable paths.
        spatial : Spatial
            Spatial data object with model boundaries and constraints.
        
        Notes
        -----
        Grid Types:
        
        Cartesian ('car'):
        - Creates regular rectangular grid
        - Uses spatial.x0, x1, y0, y1 for boundaries
        - Requires self.ncol, self.nrow for discretization
        - Automatically sets idomain based on model_boundary_poly
        - Needs upgrade to exclude inactive cells
        
        Triangular ('tri'):
        - Uses Triangle mesh generator with constraint nodes and polygons
        - Respects angle constraints (self.angle)
        - Applies area constraints from polygon definitions
        - Creates unstructured triangular elements
        
        Voronoi ('vor'):
        - First generates triangular mesh as foundation
        - Converts to Voronoi cells using dual mesh approach
        - Results in convex polygonal cells
        - Good for natural groundwater flow patterns
        
        Sets Attributes
        ---------------
        ncpl : int
            Number of cells per layer.
        cell2d : list
            Cell connectivity for DISV format.
        vertices : list
            Vertex coordinates.
        xcyc : list
            Cell center coordinates.
        xc, yc : list
            Separate x and y coordinate lists.
        vgrid : flopy.discretization.VertexGrid
            FloPy grid object for visualization.
        gi : flopy.utils.GridIntersect
            Grid intersection utility.
        idomain : ndarray
            Active cell array.
        
        Plus grid-specific attributes like tri, vor, sg, delx, dely, etc.
        """

        if self.plangrid == 'car':
            print("Creating structured grid")

            x0 = spatial.x0
            y0 = spatial.y0
            x1 = spatial.x1
            y1 = spatial.y1
            ncol = self.ncol
            nrow = self.nrow

            delx = (x1 - x0)/ncol
            dely = (y1 - y0)/nrow
            delr = delx * np.ones(ncol, dtype=float)
            delc = dely * np.ones(nrow, dtype=float)
            top  = np.ones((nrow, ncol), dtype=float)
            botm = np.zeros((1, nrow, ncol), dtype=float)

            sg = flopy.discretization.StructuredGrid(delr=delr, delc=delc, top=top, botm=botm, 
                                                    xoff = x0, yoff = y0)
            xyzcenters = sg.xyzcellcenters

            xcenters = xyzcenters[0][0]
            ycenters = [xyzcenters[1][i][0] for i in range(nrow)]
            self.xcenters, self.ycenters = xcenters, ycenters

            cell2d = []
            xcyc = [] # added 
            for n in range(nrow*ncol):
                l,r,c = sg.get_lrc(n)[0]
                xc = xcenters[c]
                yc = ycenters[r]
                iv1 = c + r * (ncol + 1)  # upper left
                iv2 = iv1 + 1
                iv3 = iv2 + ncol + 1
                iv4 = iv3 - 1
                cell2d.append([n, xc, yc, 5, iv1, iv2, iv3, iv4, iv1])
                xcyc.append((xc, yc))
            
            vertices = []
            xa = np.arange(x0, x1 + delx/2, delx)    #(x0, x1 + delx, delx)   
            ya = np.arange(y1, y0 - dely/2, -dely)
            self.xa, self.ya = xa, ya
            print(len(xa), len(ya))
            n = 0
            for j in ya:
                for i in xa:
                    vertices.append([n, i, j])
                    n+=1
            
            self.delx, self.dely = delx, dely
            self.sg = sg    
            self.cell2d = cell2d
            self.xcyc = xcyc
            self.xc, self.yc = list(zip(*self.xcyc))
            self.vertices = vertices
            self.ncpl = len(cell2d)
            
            self.vgrid = flopy.discretization.VertexGrid(vertices=vertices, 
                                                        cell2d=cell2d, 
                                                        ncpl = self.ncpl, 
                                                        nlay = 1,
                                                        crs = spatial.crs)
            self.gi = flopy.utils.GridIntersect(self.vgrid)
            
            if hasattr(spatial, 'model_boundary_poly'):
                cells_within_bd = self.gi.intersect(spatial.model_boundary_poly, geo_dataframe=True)["cellids"]
                self.idomain = np.zeros((self.ncpl))
                for icpl in cells_within_bd:
                    self.idomain[icpl] = 1
            
        if self.plangrid == 'tri':
            print("Creating triangular grid")
        
            tri = Triangle(angle    = self.angle, 
                           model_ws = project.workspace, 
                           exe_name = project.triexe, 
                           nodes    = self.nodes,
                           additional_args = ['-j','-D'])
        
            for poly in self.polygons:
                tri.add_polygon(poly[0]) 
                if poly[1] != 0: # Flag set to zero if region not required
                    tri.add_region(poly[1], 0, maximum_area = poly[2]) # Picks a point in main model
        
            tri.build(verbose=False) # Builds triangular grid
        
            self.tri = tri
            self.cell2d = tri.get_cell2d()     # cell info: id,x,y,nc,c1,c2,c3 (list of lists)
            self.vertices = tri.get_vertices()
            self.xcyc = tri.get_xcyc()
            self.xc, self.yc = list(zip(*self.xcyc))
            self.ncpl = len(self.cell2d)
            self.idomain = np.ones((self.ncpl))            
            self.vgrid = flopy.discretization.VertexGrid(vertices=self.vertices, cell2d=self.cell2d, ncpl = self.ncpl, nlay = 1)
            
        if self.plangrid == 'vor':

            print("Creating Voronoi grid")

            tri = Triangle(angle = self.angle, 
               model_ws = project.workspace, 
               exe_name = project.triexe, 
               nodes = self.nodes,
               additional_args = ['-j','-D'])

            for poly in self.polygons:
                tri.add_polygon(poly[0]) 
                if poly[1] != 0: # Flag set to zero if region not required
                    tri.add_region(poly[1], 0, maximum_area = poly[2]) # Picks a point in main model
        
            tri.build(verbose=False) # Builds triangular grid
        
            self.vor = VoronoiGrid(tri)
            self.vertices = self.vor.get_disv_gridprops()['vertices']
            self.cell2d = self.vor.get_disv_gridprops()['cell2d']

            self.ncpl = len(self.cell2d)
            self.idomain = np.ones((self.ncpl))            
            self.vgrid = flopy.discretization.VertexGrid(vertices=self.vertices, cell2d=self.cell2d, ncpl = self.ncpl, nlay = 1)
            self.xcyc = []
            for cell in self.cell2d:
                self.xcyc.append((cell[1],cell[2]))
            self.xc, self.yc = list(zip(*self.xcyc))

    def create_irregular_mesh_transect(self, crs, x0, x1, y0, y1, delc, delr):
        """
        Create a 2D transect mesh with variable column spacing.
        
        Generates a 1D mesh along a specified transect line with user-defined
        column widths. This allows for refined discretization in areas of interest
        while maintaining coarser spacing elsewhere.
        
        Parameters
        ----------
        crs : int or str
            Coordinate reference system (EPSG code or WKT string).
        x0, y0 : float
            Starting coordinates of transect line.
        x1, y1 : float
            Ending coordinates of transect line.
        delc : float
            Width of transect in perpendicular direction (m).
            This doesnt't really matter for transect models.
        delr : array_like
            Array of column widths along transect (m). Length determines ncol.
        
        Notes
        -----
        Unlike create_mesh_transect(), this method allows variable spacing:
        - Each element in delr defines width of one column
        - Total transect length is sum of delr values
        - Enables local refinement around wells or other features
        
        The method is particularly useful for:
        - Detailed modeling near pumping wells
        - Capturing sharp gradients in specific zones
        - Matching field measurement spacing
        
        Sets Attributes
        ---------------
        Similar to create_mesh_transect(), plus:
        delr : ndarray
            Variable column widths as provided.
        ncol : int
            Number of columns (= len(delr)).
        
        Examples
        --------
        >>> # Create transect with fine spacing near center
        >>> delr = [50, 50, 10, 10, 5, 5, 5, 10, 10, 50, 50]  # 11 columns
        >>> mesh = Mesh('car')
        >>> mesh.create_irregular_mesh_transect(crs=32750, x0=0, x1=sum(delr),
        ...                                     y0=0, y1=0, delc=10, delr=delr)
        """
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.crs = crs
        self.length = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        self.angrot = np.degrees(np.arctan2(self.y1 - self.y0, self.x1 - self.x0))

        # Horizontal discretisation
        self.ncol = len(delr)
        self.nrow = 1
        self.ncpl = self.ncol * self.nrow
        self.delc = delc * np.ones(self.nrow, dtype=float)
        self.delr = delr

        sg = flopy.discretization.StructuredGrid(delr=self.delr, delc=self.delc, 
                                                top=np.ones((self.nrow, self.ncol), dtype=float), 
                                                botm=np.zeros((1, self.nrow, self.ncol), dtype=float), 
                                                xoff = self.x0, yoff = self.y0, angrot = self.angrot)

        # Turn structured grid into DISV (its what I know!)
        xyzcenters = sg.xyzcellcenters
        xcenters = xyzcenters[0][0]
        ycenters = xyzcenters[1][0]
        iverts = sg.iverts
        verts = sg.verts

        cell2d = []
        xcyc = [] # added 
        for icpl in range(self.ncpl):
            xc = xcenters[icpl]
            yc = ycenters[icpl]
            iv1, iv2, iv3, iv4 = iverts[icpl]
            cell2d.append([icpl, xc, yc, 5, iv1, iv2, iv3, iv4, iv1])
            xcyc.append((xc, yc))
        
        vertices = []
        for v in range(len(verts)):
            i,j = verts[v]
            vertices.append([v, i, j]) # need to make 1 based

        self.coords = [[x0, y0], [x1, y1]]
        self.ls = LineString([[x0, y0], [x1, y1]])
        self.gdf = gpd.GeoDataFrame({'geometry': [self.ls]}, crs=self.crs)
        self.sg = sg
        self.cell2d = cell2d
        self.xcyc = xcyc
        self.xc, self.yc = list(zip(*self.xcyc))
        self.vertices = vertices
        self.xcenters, self.ycenters = xcenters, ycenters
        self.idomain = np.ones((self.ncpl))

        self.top = self.sg.top.flatten()
        self.botm = self.sg.botm.squeeze(axis=1)

        self.vgrid = flopy.discretization.VertexGrid(vertices=self.vertices, 
                                                    cell2d=self.cell2d, 
                                                    ncpl = self.ncpl, 
                                                    top = self.top,
                                                    botm = self.botm,
                                                    #top=np.ones((self.nrow, self.ncol), dtype=float), 
                                                    #botm=np.zeros((1, self.nrow, self.ncol), dtype=float), 
                                                    nlay = 1)
        self.gi = flopy.utils.GridIntersect(self.vgrid, geo_dataframe=True)

        print(f'\nTransect length: {self.length}')
        print('x0 = ', x0, ' ,y0 = ', y0)
        print('x1 = ', x1, ' ,y1 = ', y1)
        print('angrot ', self.angrot)
        print('ncol = ', self.ncol)

        # Calculate distance along trasect for each cell
        start_x = self.xc[0]
        start_y = self.yc[0]
        delr = self.delr[0]

        self.L = [] # Create a list to store distances
        for cell in range(self.ncpl):
            distance = np.sqrt((start_x - self.xc[cell])**2 + (start_y - self.yc[cell])**2 + delr/2)
            self.L.append(distance)

    def create_mesh_transect(self, crs, x0, x1, y0, y1, ncol, delc):
        """
        Create a 2D transect mesh with uniform column spacing.
        
        Generates a 1D mesh along a specified transect line for 2D cross-sectional
        modeling. The mesh is created as a thin structured grid rotated to align
        with the transect direction.
        
        Parameters
        ----------
        crs : int or str
            Coordinate reference system (EPSG code or WKT string).
        x0, y0 : float
            Starting coordinates of transect line.
        x1, y1 : float
            Ending coordinates of transect line.
        ncol : int
            Number of columns (cells) along transect.
        delc : float
            Width of transect in perpendicular direction (m).
        
        Notes
        -----
        The method:
        1. Calculates transect length and rotation angle
        2. Creates uniform column spacing (delr = length/ncol)
        3. Generates structured grid aligned with transect
        4. Converts to DISV format for compatibility
        5. Calculates distance along transect for each cell
        
        The resulting mesh is suitable for:
        - 2D cross-sectional flow modeling
        - Transient transport simulations
        - Detailed analysis along specific flow paths
        
        Sets Attributes
        ---------------
        length : float
            Total length of transect (m).
        angrot : float
            Rotation angle from east (degrees).
        ncol, nrow : int
            Number of columns and rows (nrow=1).
        ncpl : int
            Number of cells per layer (= ncol).
        delr, delc : float or ndarray
            Cell spacing in row and column directions.
        L : list
            Distance along transect for each cell center.
        coords : list
            Start and end coordinates of transect.
        ls : LineString
            Shapely LineString geometry of transect.
        gdf : GeoDataFrame
            GeoPandas DataFrame containing transect geometry.
        
        Examples
        --------
        >>> # Create 500m transect with 25 cells
        >>> mesh = Mesh('car')
        >>> mesh.create_mesh_transect(crs=32750, x0=700000, x1=700500, 
        ...                          y0=6200000, y1=6200000, ncol=25, delc=10)
        """
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.crs = crs
        self.length = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        self.angrot = np.degrees(np.arctan2(self.y1 - self.y0, self.x1 - self.x0))

        # Horizontal discretisation
        self.ncol = ncol
        delr = self.length / self.ncol
        self.nrow = 1
        self.ncpl = self.ncol * self.nrow
        self.delr = delr * np.ones(self.ncol, dtype=float)
        self.delc = delc * np.ones(self.nrow, dtype=float)

        sg = flopy.discretization.StructuredGrid(delr=self.delr, delc=self.delc, 
                                                 top=np.ones((self.nrow, self.ncol), dtype=float), 
                                                 botm=np.zeros((1, self.nrow, self.ncol), dtype=float), 
                                                 xoff = self.x0, yoff = self.y0, angrot = self.angrot)

        # Turn structured grid into DISV (its what I know!)
        xyzcenters = sg.xyzcellcenters
        xcenters = xyzcenters[0][0]
        ycenters = xyzcenters[1][0]
        iverts = sg.iverts
        verts = sg.verts

        cell2d = []
        xcyc = [] # added 
        for icpl in range(self.ncpl):
            xc = xcenters[icpl]
            yc = ycenters[icpl]
            iv1, iv2, iv3, iv4 = iverts[icpl]
            cell2d.append([icpl, xc, yc, 5, iv1, iv2, iv3, iv4, iv1])
            xcyc.append((xc, yc))
        
        vertices = []
        for v in range(len(verts)):
            i,j = verts[v]
            vertices.append([v, i, j]) # need to make 1 based

        self.coords = [[x0, y0], [x1, y1]]
        self.ls = LineString([[x0, y0], [x1, y1]])
        self.gdf = gpd.GeoDataFrame({'geometry': [self.ls]}, crs=self.crs)
        self.sg = sg
        self.cell2d = cell2d
        self.xcyc = xcyc
        self.xc, self.yc = list(zip(*self.xcyc))
        self.vertices = vertices
        self.xcenters, self.ycenters = xcenters, ycenters
        self.idomain = np.ones((self.ncpl))

        self.top = self.sg.top.flatten()
        self.botm = self.sg.botm.squeeze(axis=1)

        self.vgrid = flopy.discretization.VertexGrid(vertices=self.vertices, 
                                                     cell2d=self.cell2d, 
                                                     ncpl = self.ncpl, 
                                                     top = self.top,
                                                     botm = self.botm,
                                                     #top=np.ones((self.nrow, self.ncol), dtype=float), 
                                                     #botm=np.zeros((1, self.nrow, self.ncol), dtype=float), 
                                                     nlay = 1)
        self.gi = flopy.utils.GridIntersect(self.vgrid, geo_dataframe=True)

        print(f'\nTransect length: {self.length}')
        print('x0 = ', x0, ' ,y0 = ', y0)
        print('x1 = ', x1, ' ,y1 = ', y1)
        print('angrot ', self.angrot)
        print('ncol = ', self.ncol)

        # Calculate distance along trasect for each cell
        start_x = self.xc[0]
        start_y = self.yc[0]
        delr = self.delr[0]

        self.L = [] # Create a list to store distances
        for cell in range(self.ncpl):
            distance = np.sqrt((start_x - self.xc[cell])**2 + (start_y - self.yc[cell])**2 + delr/2)
            self.L.append(distance)
    
    def locate_special_cells(self, spatial, threshold = 1.0):
        """
        Identify and classify mesh cells requiring special treatment.
        
        Locates cells that intersect with important features like wells, boundaries,
        and constraint polygons. These cells are flagged for special handling in
        boundary condition and source/sink term assignment.
        
        Parameters
        ----------
        spatial : Spatial
            Spatial data object containing feature geometries.
        threshold : float, optional
            Minimum fraction of cell area that must overlap with a feature
            for the cell to be classified as 'special' (default: 1.0).
        
        Notes
        -----
        Supported Feature Types:
        
        - 'obs': Observation well points
        - 'wel': Pumping well points  
        - 'chd': Constant head boundaries (lines)
        - 'ghb': General head boundaries (lines)
        - 'poly': Single polygon features
        - 'multipoly': Multi-polygon features 
        
        For each feature type, the method:
        1. Finds intersecting cells using grid intersection
        2. Assigns unique flag numbers for identification
        3. Stores cell lists for each boundary condition type
        4. Creates cell type labels for visualization
        
        The threshold parameter is particularly useful for polygon features
        where you only want cells with significant overlap to be flagged.
        
        Sets Attributes
        ---------------
        obs_cells, wel_cells, chd_cells, ghb_cells : list
            Lists of cell indices for each boundary condition type.
        lak_cells, drn_cells, poly_cells : list
            Additional cell lists for other feature types.
        ibd : ndarray
            Special cell flag array with unique ID for each feature.
        cell_type : list
            Descriptive labels for each flag value.
        
        Dynamic attributes are also created:
        chd_{subgroup}_cells, ghb_{subgroup}_cells, etc.
            Cell lists for specific feature subgroups.
        
        Examples
        --------
        >>> # Define special cells with observation and pumping wells
        >>> special_cells = {
        ...     'obs': ['monitoring'],
        ...     'wel': ['production'],
        ...     'chd': ['west', 'east']
        ... }
        >>> mesh = Mesh('tri', special_cells=special_cells)
        >>> mesh.locate_special_cells(spatial, threshold=0.8)
        """
        
        self.obs_cells = [] 
        self.wel_cells = [] 
        self.chd_cells = [] 
        self.ghb_cells = [] 
        self.lak_cells = [] 
        self.drn_cells = [] 
        self.poly_cells = []

        self.gi = flopy.utils.GridIntersect(self.vgrid)
        self.ibd = np.zeros(self.ncpl, dtype=int) # empty array top shade important cells
        flag = 1 # id of special cell
        self.cell_type = ['regular cell']
        
        for group in self.special_cells:
            
            subgroups = self.special_cells[group]
            print('Group = ', group, subgroups)
            print('flag =', flag)

            #for attribute, value in flowmodel.data.__dict__.items(): print(attribute)
            #for key, value in d.items():
            #print(f"{key}: {value}")
            if group == 'obs':
                for i, subgroup in enumerate(subgroups): # e.g. for 'west' in chd
                    self.cell_type.append(f'{group} - {subgroup}')
                    points = [Point(xy) for xy in spatial.xyobsbores]
                    for point in points:
                        cell = self.gi.intersect(point, geo_dataframe=True)["cellids"][0]
                        self.ibd[cell] = flag
                        self.obs_cells.append(cell)
                    flag += 1
                    
                
            if group == 'wel':
                for i, subgroup in enumerate(subgroups): # e.g. for 'west' in chd
                    self.cell_type.append(f'{group} - {subgroup}')
                    points = [Point(xy) for xy in spatial.xypumpbores]
                    for point in points:
                        cell = self.gi.intersect(point, geo_dataframe=True)["cellids"][0]
                        self.ibd[cell] = flag
                        self.wel_cells.append(cell)
                    flag += 1
                    
               
            if group == 'chd':    
                for i, subgroup in enumerate(subgroups): # e.g. for 'west' in chd    
                    self.cell_type.append(f'{group} - {subgroup}')
                    att_name = f"chd_{subgroup}_ls"
                    ls = getattr(spatial, att_name)
                    
                    cells = self.gi.intersects(ls, geo_dataframe=True)["cellids"]

                    att_name = f"chd_{subgroup}_cells"
                    setattr(self, att_name, cells)
                    print(att_name, cells)
                    
                    for cell in cells:
                        self.chd_cells.append(cell)
                        self.ibd[cell] = flag
                    flag += 1

            if group == 'ghb':    
                for i, subgroup in enumerate(subgroups): # e.g. for 'west' in chd    
                    self.cell_type.append(f'{group} - {subgroup}')
                    att_name = f"ghb_{subgroup}_ls"
                    ls = getattr(spatial, att_name)
                    
                    cells = self.gi.intersects(ls, geo_dataframe=True)["cellids"]

                    att_name = f"ghb_{subgroup}_cells"
                    setattr(self, att_name, cells)
                    
                    for cell in cells:
                        self.ghb_cells.append(cell)
                        self.ibd[cell] = flag
                    flag += 1
                             
            if group == 'poly':    
                for i, subgroup in enumerate(subgroups): # e.g. for 'river1' in poly   
                    self.cell_type.append(f'{group} - {subgroup}')
                    att_name = f"{subgroup}_poly"
                    poly = getattr(spatial, att_name)
                    result = self.gi.intersect(poly, geo_dataframe=True)
                    cells = result.cellids

                    for cell in cells:
                        self.ibd[cell] = flag
                    
                    att_name = f"poly_{subgroup}_cells"
                    print(att_name)
                    setattr(self, att_name, cells)
                    flag += 1
            
            if group == 'multipoly':
                print(dir(self.vgrid))
                def path_to_polygon(path):
                    verts = path.vertices # The mesh is not a gdf at this point, but it needs to be to interact with other gdfs as below - create polygon
                    return Polygon(verts)
                
                for i, subgroup in enumerate(subgroups):
                    self.cell_type.append(f'{group} - {subgroup}') #for each individual shape within the veg group?
                    att_name = f"{subgroup}_multipoly"
                    multipoly = getattr(spatial, att_name)

                    apply_threshold = getattr(self, "threshold", threshold) #set a threshold so not EVERY cell that has even a little bit of veg will be 'special'
                    intersecting_cells = []

                    for idx, cell_path in enumerate(self.vgrid.map_polygons):
                        cell_geom = path_to_polygon(cell_path)
                        intersection = cell_geom.intersection(multipoly)

                        if not intersection.is_empty:
                            overlap_area = intersection.area
                            cell_area = cell_geom.area
                            if overlap_area / cell_area >= apply_threshold:
                                intersecting_cells.append(idx)  # or use row.cellid if available

                    for cell in intersecting_cells:
                        self.ibd[cell] = flag
                    
                    att_name = f"{subgroup}_cells"
                    print(att_name)
                    setattr(self, att_name, cells)
                    flag += 1
            
    def plot_cell2d(self, spatial, features = None, xlim = None, ylim = None, 
                    nodes = False, labels = False, fname = '../figures/mesh.png'):
        """
        Plot the 2D mesh with optional overlay of spatial features.
        
        Creates a plan view visualization of the computational mesh showing
        cell boundaries, centers, and optionally various spatial features
        like wells, boundaries, and geological structures.
        
        Parameters
        ----------
        spatial : Spatial
            Spatial data object containing feature geometries and geodataframes.
        features : list of str, optional
            Features to overlay on mesh plot. Options include:
            - 'geo': Geological borehole locations
            - 'obs': Observation well locations
            - 'wel': Pumping well locations  
            - 'fault': Fault lines
            - 'river': River polygons
        xlim, ylim : tuple, optional
            Plot extent limits as (min, max) tuples.
        nodes : bool, optional
            Whether to plot constraint nodes used in mesh generation (default: False).
        labels : bool, optional
            Whether to add text labels to well locations (default: False).
        fname : str, optional
            Filename to save the plot (default: '../figures/mesh.png').
        
        Notes
        -----
        The plot includes:
        - Mesh grid lines and cell centers (green dots)
        - Model boundary (black solid line)
        - Inner boundary (black dashed line)
        - Selected features with appropriate symbols and colors
        
        Feature Styling:
        - Geological bores: Green circles
        - Observation wells: Dark blue circles
        - Pumping wells: Red circles (larger)
        - Faults: Red lines
        - Rivers: Blue polygons with transparency
        
        Examples
        --------
        >>> # Plot mesh with wells and rivers
        >>> mesh.plot_cell2d(spatial, features=['obs', 'wel', 'river'], 
        ...                  labels=True, xlim=(700000, 710000))
        """
        
        fig = plt.figure(figsize=(7,7))
        ax = plt.subplot(1, 1, 1, aspect='auto')
        if xlim: ax.set_xlim(xlim) 
        if ylim: ax.set_ylim(ylim) 
        ax.set_title('Number cells in plan (ncpl): ' + str(len(self.cell2d)))
    
        pmv = flopy.plot.PlotMapView(ax = ax, modelgrid=self.vgrid)
        pmv.plot_grid(color = 'gray', lw = 0.8)

        for i in self.xcyc: 
                ax.plot(i[0], i[1], 'o', color = 'green', ms = 0.5)    
        if nodes:
            for i in self.nodes: 
                ax.plot(i[0], i[1], 'o', ms = 1, color = 'black')
        x, y = spatial.model_boundary_poly.exterior.xy
        ax.plot(x, y, '-o', ms = 2, lw = 1, color='black')
        x, y = spatial.inner_boundary_poly.exterior.xy
        ax.plot(x, y, '-o', ms = 2, lw = 0.5, color='black')
            
        if 'geo' in features:
            spatial.geobore_gdf.plot(ax=ax, markersize = 7, color = 'green', zorder=2)
            if labels:
                for x, y, label in zip(spatial.geobore_gdf.geometry.x, spatial.geobore_gdf.geometry.y, spatial.obsbore_gdf.ID):
                    ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 8, textcoords="offset points")

        if 'obs' in features:
            spatial.obsbore_gdf.plot(ax=ax, markersize = 7, color = 'darkblue', zorder=2)
            if labels:
                for x, y, label in zip(spatial.obsbore_gdf.geometry.x, spatial.obsbore_gdf.geometry.y, spatial.obsbore_gdf.ID):
                    ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 8, textcoords="offset points")

        if 'wel' in features:
            spatial.pumpbore_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2) 
            if labels:
                for x, y, label in zip(spatial.pumpbore_gdf.geometry.x, spatial.pumpbore_gdf.geometry.y, spatial.pumpbore_gdf.ID):
                    ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 8, textcoords="offset points")
        
        if 'fault' in features:
            spatial.faults_gdf.plot(ax=ax, color = 'red', zorder=2)

        if 'river' in features:
            spatial.river_gdf.plot(ax=ax, color = 'blue', alpha = 0.4, zorder=2)

        plt.savefig(fname)
            
    def plot_feature_cells(self, spatial, xlim = None, ylim = None, 
                           plot_grid = False, 
                           fname = '../figures/special_cells.png'):
        """
        Plot mesh cells colored by special feature classification.
        
        Creates a visualization showing which cells have been identified as
        'special' for boundary conditions or other purposes, with different
        colors for each feature type and a legend explaining the classification.
        
        Parameters
        ----------
        spatial : Spatial
            Spatial data object containing feature geometries.
        xlim, ylim : tuple, optional
            Plot extent limits as (min, max) tuples.
        plot_grid : bool, optional
            Whether to overlay mesh grid lines (default: False).
        fname : str, optional
            Filename to save the plot (default: '../figures/special_cells.png').
        
        Notes
        -----
        The visualization shows:
        - Cells colored by their ibd (special cell) classification
        - Different colors for each boundary condition type
        - Wells and boundaries overlaid with appropriate symbols
        - Colorbar with feature type labels
        - Only active cells (idomain=1) are colored
        
        This plot is useful for:
        - Verifying correct identification of boundary cells
        - Quality control of mesh-feature intersection
        - Understanding spatial distribution of boundary conditions
        - Debugging boundary condition setup
        
        The plot uses matplotlib's 'tab20' colormap to provide distinct
        colors for up to 20 different feature types.
        
        The plot is automatically saved to '../figures/special_cells.png'.
        
        Examples
        --------
        >>> # Plot special cells with grid overlay
        >>> mesh.plot_feature_cells(spatial, plot_grid=True, 
        ...                        xlim=(700000, 710000))
        """
        
        fig = plt.figure(figsize=(7,5))
        spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 0.05], wspace=0.2)

        ax = fig.add_subplot(spec[0], aspect="equal") #plt.subplot(1, 1, 1, aspect="auto")
        if xlim: ax.set_xlim(xlim) 
        if ylim: ax.set_ylim(ylim) 
        ax.set_title('Special cells')
            
        pmv = flopy.plot.PlotMapView(ax = ax, modelgrid=self.vgrid)
        
        if plot_grid: 
            pmv.plot_grid(color = 'black', lw = 0.4)
        else:
            pmv.plot_grid(visible=False)

        unique_ibd = np.unique(self.ibd)
        print(unique_ibd)
        bounds = np.append(unique_ibd, unique_ibd[-1]+1)
        cmap = plt.cm.tab20
        #cmap = plt.cm.get_cmap('tab20', len(unique_ibd))  # 'tab20' provides 20 distinct colors
        #colors = [cmap(i) for i in range(cmap.N)]
        norm = BoundaryNorm(bounds, cmap.N, extend='neither')
        
        mask = self.idomain == 0
        ma = np.ma.masked_where(mask, self.ibd)
        p = pmv.plot_array(ma, alpha = 0.6, cmap = cmap, norm = norm)
        
        
        for group in self.special_cells:
            
            if group == 'obs':
                for i in range(len(spatial.obsbore_gdf)):
                    x,y = spatial.obsbore_gdf.geometry.iloc[i].xy
                    ax.plot(x, y, '-o', ms = 2, lw = 1, color='blue') ###########
                #for cell in self.obs_cells:  
                #    ax.plot(self.cell2d[cell][1], self.cell2d[cell][2], "o", color = 'black', ms = 1)############
            
            if group == 'wel':
                for i in range(len(spatial.pumpbore_gdf)):
                    x,y = spatial.pumpbore_gdf.geometry.iloc[i].xy
                    ax.plot(x, y, '-o', ms = 2, lw = 1, color='red')
                #for cell in self.wel_cells:
                #    ax.plot(self.cell2d[cell][1], self.cell2d[cell][2], "o", color = 'blue', ms = 2)

            if group == 'chd':
                for cell in self.chd_cells:
                    ax.plot(self.cell2d[cell][1], self.cell2d[cell][2], "o", color = 'red', ms = 1)

            if group == 'ghb':
                for cell in self.ghb_cells:
                    ax.plot(self.cell2d[cell][1], self.cell2d[cell][2], "o", color = 'red', ms = 1)

            #if group == 'zone':
            #    for subgroup in self.special_cells['zone']: # subgroup must be a poly
            #        x, y = spatial.subgroup.exterior.xy
            #        ax.plot(x, y, '-o', ms = 2, lw = 0.5, color='green') 
        
        print(np.unique(self.ibd))
        # Colorbar
        cbar_ax = fig.add_subplot(spec[1])
        cbar = fig.colorbar(p, cax=cbar_ax, ticks=np.unique(self.ibd)+0.5, shrink = 0.1)  # Center tick labels
        cbar.ax.set_yticklabels(self.cell_type) # Custom tick labels

        plt.savefig(fname)

    def plot_surface_array(self, array, structuralmodel, 
                           plot_data = False, 
                           vmin = None, vmax = None, 
                           plot_grid = False, 
                           levels = None, title = None):
        """
        Plot a 2D array over the mesh surface with geological context.
        
        Creates a plan view visualization of any 2D array (e.g., hydraulic head,
        surface elevation, lithology) with optional contours and geological data
        overlay. Designed specifically for geological modeling applications.
        
        Parameters
        ----------
        array : array_like
            2D array with values for each mesh cell (length = ncpl).
        structuralmodel : StructuralModel
            Structural model object containing geological data.
        plot_data : bool, optional
            Whether to overlay raw geological data points (default: False).
        vmin, vmax : float, optional
            Color scale limits (default: None, auto-scale).
        plot_grid : bool, optional
            Whether to show mesh grid lines (default: False).
        levels : array_like, optional
            Contour levels to draw (default: None, no contours).
        title : str, optional
            Plot title (default: None).
        
        Notes
        -----
        This method is optimized for geological applications and includes:
        - Color-filled array visualization
        - Optional contour lines with specified levels
        - Overlay of original geological data points
        - Proper handling of structural model data formats
        
        Useful for plotting:
        - Surface elevation models
        - Lithological classifications
        - Hydraulic head distributions
        - Model validation against field data
        
        Examples
        --------
        >>> # Plot surface elevation with 10m contours
        >>> levels = np.arange(0, 200, 10)
        >>> mesh.plot_surface_array(elevation_array, struct_model,
        ...                        plot_data=True, levels=levels,
        ...                        title='Ground Surface Elevation')
        """
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot()
        if title: ax.set_title(title)
        
        pmv = flopy.plot.PlotMapView(modelgrid=self.vgrid)
        t = pmv.plot_array(array, vmin = vmin, vmax = vmax)
        cbar = plt.colorbar(t, shrink = 0.5)  
        cg = pmv.contour_array(array, levels=levels, linewidths=0.8, colors="0.75")

        if plot_grid == False:
            pmv.plot_grid(visible = False)
        
        if plot_data:# Plot raw data points
            df = structuralmodel.data
            ax.plot(df.X, df.Y, 'o', ms = 2, color = 'red')
    
    def plot_array(self, array, 
                   vmin = None, 
                   vmax = None, 
                   levels = None, 
                   title = None, 
                   xlim = None, ylim = None,
                   xy = None):
        """
        Plot a 2D array over the mesh with optional point overlay.
        
        Creates a general-purpose plan view visualization of any 2D array
        with customizable styling and optional point data overlay.
        
        Parameters
        ----------
        array : array_like
            2D array with values for each mesh cell (length = ncpl).
        vmin, vmax : float, optional
            Color scale limits (default: None, auto-scale).
        levels : array_like, optional
            Contour levels to draw (default: None, no contours).
        title : str, optional
            Plot title (default: None).
        xlim, ylim : tuple, optional
            Plot extent limits as (min, max) tuples.
        xy : tuple of array_like, optional
            Point coordinates to overlay as (x_coords, y_coords).
        
        Notes
        -----
        This is a simplified, general-purpose plotting method that:
        - Shows color-filled array values
        - Optionally overlays point data in red
        - Includes colorbar for scale reference
        - Allows custom plot extents
        
        Unlike plot_surface_array(), this method doesn't include
        geological-specific features but offers more flexible
        point data overlay capabilities.
        
        Useful for:
        - Quick visualization of model results
        - Plotting hydraulic head, concentration, or velocity
        - Overlaying measurement locations
        - General array visualization tasks
        
        Examples
        --------
        >>> # Plot hydraulic head with well locations
        >>> well_x = [700000, 705000, 710000]
        >>> well_y = [6200000, 6205000, 6210000]
        >>> mesh.plot_array(head_array, title='Hydraulic Head (m)',
        ...                 xy=(well_x, well_y), vmin=50, vmax=150)
        """
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot()
        if title: ax.set_title(title)
        pmv = flopy.plot.PlotMapView(modelgrid=self.vgrid)
        t = pmv.plot_array(array, vmin = vmin, vmax = vmax)
        cbar = plt.colorbar(t, shrink = 0.5)  
        #cg = pmv.contour_array(array, levels=levels, linewidths=0.8, colors="0.75")
        if xy:# Plot points
            ax.plot(xy[0], xy[1], 'o', ms = 2, color = 'black')
        if xlim: ax.set_xlim(xlim) 
        if ylim: ax.set_ylim(ylim) 