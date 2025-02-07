import flopy
from shapely.geometry import Point
import geopandas as gpd

class SurfaceRefinement:
    def __init__(self, structuralmodel, nrow, ncol):
        self.classlabel = "SurfaceRefinementBaseClass"
        self.nrow = nrow
        self.ncol = ncol

    def create_surf_lith_raster(self, project, spatial, structuralmodel):    
        # Pick max and min ground levels
        ground_entries = structuralmodel.data[structuralmodel.data['lithcode'] == 'Ground']
        max_gl, min_gl = max(ground_entries.Z), min(ground_entries.Z)
        print('Max ground level = ', max_gl)
        print('Min ground level = ', min_gl)
        z0, z1 = min_gl-2, max_gl+2
        structuralmodel.max_RL, structuralmodel.min_RL = max_gl, min_gl
        
        # Create a structured mesh to detect lithological interfaces
        from loopflopy.mesh import Mesh
        mesh = Mesh(plangrid = 'car') 
        mesh.ncol, mesh.nrow = self.ncol, self.nrow
        mesh.create_mesh(project, spatial)
        print('number of cells in plan = ', mesh.ncpl)

        # Create a geomodel (near surface) to be able to find surface lithology
        scenario = 'surf_lith'
        vertgrid = 'con'    # 'vox', 'con' or 'con2'
        from loopflopy.geomodel import Geomodel
        geomodel = Geomodel(scenario, vertgrid, z0, z1, nls = 1, res = 2)#, max_thick = 100. * np.ones((7)))

        geomodel.create_lith_dis_arrays(mesh, structuralmodel)
        geomodel.vgrid = flopy.discretization.VertexGrid(vertices=mesh.vertices, cell2d=mesh.cell2d, ncpl = mesh.ncpl, top = geomodel.top_geo, botm = geomodel.botm)
        geomodel.get_surface_lith()

        # Add refinement nodes at surface lithoogy interface
        self.raster = geomodel.surf_lith.reshape((mesh.nrow, mesh.ncol))
        self.geomodel = geomodel
        self.mesh = mesh

    # Function to find interfaces and generate nodes
    def generate_interface_nodes(self, spatial):
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
        spatial.interface_nodes = list(zip(gdf.geometry.x, gdf.geometry.y)) # Save the nodes as a list of tuples

        return gdf
    
    def plot_surface_refinement(self, spatial, structuralmodel, y0, y1):
        self.geomodel.geomodel_plan_lith(spatial, self.mesh, structuralmodel, y0 = y0, y1 = y1)
        self.geomodel.geomodel_transect_lith(structuralmodel, spatial, y0 = y0, y1 = y1)#, z0 = -900, z1 = -2000) 

