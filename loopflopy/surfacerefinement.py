import flopy
from shapely.geometry import Point, LineString
import geopandas as gpd
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from loopflopy.mesh_routines import create_structured_mesh

class SurfaceRefinement:
    def __init__(self, structuralmodel, nrow, ncol):
        self.classlabel = "SurfaceRefinementBaseClass"
        self.nrow = nrow
        self.ncol = ncol
        self.bbox = structuralmodel.bbox

    def create_surf_lith_raster(self, structuralmodel):    
        # Pick max and min ground levels
        ground_entries = structuralmodel.data[structuralmodel.data['lithcode'] == 'Ground']
        max_gl, min_gl = max(ground_entries.Z), min(ground_entries.Z)
        print('Max ground level = ', max_gl)
        print('Min ground level = ', min_gl)
        z0, z1 = min_gl-2, max_gl+2
        structuralmodel.max_RL, structuralmodel.min_RL = max_gl, min_gl
        
        # Create a structured mesh to detect lithological interfaces
        #mesh  = create_mesh(self.bbox, self.ncol, self.nrow)
        mesh = create_structured_mesh(self.bbox, self.ncol, self.nrow)

        print('number of cells in plan = ', mesh.ncpl)

        # Create a geomodel (near surface) to be able to find surface lithology
        scenario = 'surf_lith'
        vertgrid = 'con'    # 'vox', 'con' or 'con2'
        from loopflopy.geomodel import Geomodel
        geomodel = Geomodel(scenario, vertgrid, z0, z1, nls = 1, res = 1)#, max_thick = 100. * np.ones((7)))

        geomodel.evaluate_structuralmodel(mesh, structuralmodel)
        surface = structuralmodel.topo
        geomodel.create_model_layers(mesh, structuralmodel, surface)
        #geomodel.create_lith_dis_arrays(mesh, structuralmodel)
        geomodel.vgrid = flopy.discretization.VertexGrid(vertices=mesh.vertices, cell2d=mesh.cell2d, ncpl = mesh.ncpl, 
                                                         top = geomodel.top_geo, botm = geomodel.botm)
        geomodel.get_surface_lith()

        # Add refinement nodes at surface lithoogy interface
        self.raster = geomodel.surf_lith.reshape((mesh.nrow, mesh.ncol))
        self.geomodel = geomodel
        self.mesh = mesh

    # Function to find interfaces and generate nodes
    '''def generate_interface_nodes(self, spatial):
        raster = self.raster

        nodes = []

        for i in range(self.nrow - 1):
            for j in range(self.ncol - 1):
                if raster[i, j] != raster[i, j + 1]:
                    node = Point(self.mesh.xcenters[j] + 0.5 * self.mesh.delx, self.mesh.ycenters[i] - 0.5 * self.mesh.dely)
                    if node.within(spatial.model_boundary_poly.buffer(-2 * spatial.boundary_buff)): # So that no nodes too close to model boundary 
                        nodes.append(node)
                if raster[i, j] != raster[i + 1, j]:
                    node = Point(self.mesh.xcenters[j] + 0.5 * self.mesh.delx, self.mesh.ycenters[i] - 0.5 * self.mesh.dely)
                    if node.within(spatial.model_boundary_poly.buffer(-2 * spatial.boundary_buff)): # So that no nodes too close to model boundary 
                        nodes.append(node)
                if raster[i, j] != raster[i + 1, j + 1]:
                    node = Point(self.mesh.xcenters[j] + 0.5 * self.mesh.delx, self.mesh.ycenters[i] - 0.5 * self.mesh.dely)
                    if node.within(spatial.model_boundary_poly.buffer(-2 * spatial.boundary_buff)): # So that no nodes too close to model boundary 
                        nodes.append(node)
        self.nodes = nodes
        gdf = gpd.GeoDataFrame(crs = spatial.epsg, geometry=nodes) # Create a GeoDataFrame from the nodes
        gdf.to_file("../modelfiles/interface_nodes.shp") # Save the nodes as a shapefile
        spatial.interface_nodes = list(zip(gdf.geometry.x, gdf.geometry.y)) # Save the nodes as a list of tuples'''
    
    def plot_surface_refinement(self, spatial, structuralmodel, y0, y1):
        self.geomodel.geomodel_plan_lith(spatial, self.mesh, structuralmodel, y0 = y0, y1 = y1)
        self.geomodel.geomodel_transect_lith(structuralmodel, spatial, y0 = y0, y1 = y1)#, z0 = -900, z1 = -2000) 


    def array_intersection(self, project, structuralmodel, array1, array2, plot_datapoints = False, **kwargs): # This contours using the DEM with a surface

            thickness = array1 - array2

            fig = plt.figure(figsize = (10, 6))
            ax = plt.subplot(111)
            ax.set_aspect('equal')
            ax.set_title('surface geology', size = 10)
            ax.set_xlabel('x (m)', size = 10)
            ax.set_ylabel('y (m)', size = 10)

            if plot_datapoints:
                ax.plot(structuralmodel.data.X, structuralmodel.data.Y, 'o', ms = 1, color = 'red')
            
            mapview = flopy.plot.PlotMapView(modelgrid=self.geomodel.vgrid, layer = 0, ax = ax)
            
            plan = mapview.plot_array(thickness, 
                                      cmap='Spectral',
                                      alpha=0.8, ax = ax)
            
            # Contours
            X, Y = self.mesh.xcenters, self.mesh.ycenters
            
            Z = thickness.reshape((self.mesh.nrow, self.mesh.ncol))
            levels = np.arange(0, 999999)
            contour = ax.contour(X, Y, Z, 
                                 levels = [0.], 
                                 extend = 'both', colors = 'Black', 
                                 linewidths=1., linestyles = 'solid')
            
            cbar = plt.colorbar(plan, shrink = 1.0)
            plt.tight_layout()  
            plt.show()

            # Extract contour lines
            contour_lines = []
            for collection in contour.collections:
                for path in collection.get_paths():
                    v = path.vertices
                    contour_lines.append(LineString(v))

            #save_contour_lines_to_shapefile
            gdf = gpd.GeoDataFrame(geometry=contour_lines, crs = project.crs)
            gdf.to_file('../data/data_shp/intersection_contour.shp', driver='ESRI Shapefile')
            self.gdf = gdf

    def surface_contours(self, project, structuralmodel, plot_datapoints = False, **kwargs): # This contours surface lithology using the geomodel's surface lithology

            fig = plt.figure(figsize = (10, 6))
            ax = plt.subplot(111)
            ax.set_aspect('equal')
            ax.set_title('surface geology', size = 10)
            ax.set_xlabel('x (m)', size = 10)
            ax.set_ylabel('y (m)', size = 10)

            if plot_datapoints:
                ax.plot(structuralmodel.data.X, structuralmodel.data.Y, 'o', ms = 1, color = 'red')
            
            mapview = flopy.plot.PlotMapView(modelgrid=self.geomodel.vgrid, layer = 0, ax = ax)
            
            plan = mapview.plot_array(self.geomodel.surf_lith, 
                                      cmap=structuralmodel.cmap, 
                                      norm = structuralmodel.norm, 
                                      alpha=0.8, ax = ax)

            labels = structuralmodel.strat_names[1:]
            ticks = [i for i in np.arange(0,len(labels))]
            boundaries = np.arange(-1,len(labels),1)+0.5       
            
            # Contours
            X, Y = self.mesh.xcenters, self.mesh.ycenters
            Z = self.geomodel.surf_lith.reshape((self.mesh.nrow, self.mesh.ncol))
            levels = np.arange(0, self.geomodel.nlg+1, 1)
            contour = ax.contour(X, Y, Z, 
                                 levels = levels, 
                                 extend = 'both', colors = 'Black', 
                                 linewidths=1., linestyles = 'solid')
            
            cbar = plt.colorbar(plan,
                        boundaries = boundaries,
                        shrink = 1.0)
            cbar.ax.set_yticks(ticks = ticks, labels = labels, size = 8, verticalalignment = 'center')   
            plt.tight_layout()  
            plt.show()

            # Extract contour lines
            contour_lines = []
            for collection in contour.collections:
                for path in collection.get_paths():
                    v = path.vertices
                    contour_lines.append(LineString(v))

            #save_contour_lines_to_shapefile
            gdf = gpd.GeoDataFrame(geometry=contour_lines, crs = project.crs)
            gdf.to_file('../data/data_shp/surface_contours.shp', driver='ESRI Shapefile')
            self.gdf = gdf
