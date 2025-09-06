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
    #
    #           theta = np.linspace(0, 2 * np.pi, 11)
    #           for i in spatial.xypumpbores:   
    #               X = i[0] + self.radius1 * np.cos(theta)
    #               Y = i[1] + self.radius1 * np.sin(theta)    
    #               vertices1 = [(x_val, y_val) for x_val, y_val in zip(X, Y)]
    #               X = i[0] + self.radius2 * np.cos(theta)
    #               Y = i[1] + self.radius2 * np.sin(theta)    
    #               vertices2 = [(x_val, y_val) for x_val, y_val in zip(X, Y)]
    #               self.welnodes.append(vertices1)
    #               self.welnodes.append(vertices2)
    #               #self.welnodes2.append(vertices2)
                        
            
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
                cells_within_bd = self.gi.intersect(spatial.model_boundary_poly)["cellids"]
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

            # Check northernmost cell centre not on the boundary, if so, shift down
            yc = [row[2] for row in self.cell2d]
            max_yc = np.max(yc)           
            bounds = spatial.model_boundary_poly.bounds # Get the bounds of the polygon (minx, miny, maxx, maxy)
            max_ybound = bounds[3]  # maxy is at index 3

            if max_yc >= max_ybound:
                index = np.where(yc >= max_yc)[0][0]
                print(f'Cell {index} is on the model boundary. Moving down by 0.1m')
                self.cell2d[index][2] -= 0.1  

            self.ncpl = len(self.cell2d)
            self.idomain = np.ones((self.ncpl))            
            self.vgrid = flopy.discretization.VertexGrid(vertices=self.vertices, cell2d=self.cell2d, ncpl = self.ncpl, nlay = 1)
            self.xcyc = []
            for cell in self.cell2d:
                self.xcyc.append((cell[1],cell[2]))
            self.xc, self.yc = list(zip(*self.xcyc))

    '''def create_mesh_transect(self, x0, x1, y0, y1, delr, delc): # delr is a list of column widths
        if self.plangrid == 'transect':

            self.ncol = len(delr) # number of columns in transect
            self.nrow = 1
            self.ncpl = self.ncol * self.nrow
            delr = np.array(delr)
            delc = delc * np.ones(self.nrow, dtype=float)
            top  = np.ones((self.nrow, self.ncol), dtype=float)
            botm = np.zeros((1, self.nrow, self.ncol), dtype=float)
            
            angrot = np.degrees(np.arctan((y0 - y1)/(x0 - x1)))
            print('angrot ', angrot)   
            sg = flopy.discretization.StructuredGrid(delr=delr, delc=delc, top=top, botm=botm, 
                                                    xoff = x0, yoff = y0, angrot = angrot)
                                                    
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

            self.sg = sg    
            self.cell2d = cell2d
            self.xcyc = xcyc
            self.xc, self.yc = list(zip(*self.xcyc))
            self.vertices = vertices
            self.xyzcenters = xyzcenters
            self.xcenters, self.ycenters = xcenters, ycenters
            self.idomain = np.ones((self.ncpl))
            self.vgrid = flopy.discretization.VertexGrid(vertices=self.vertices, cell2d=self.cell2d, ncpl = self.ncpl, nlay = 1)
            self.gi = flopy.utils.GridIntersect(self.vgrid)'''

    def create_irregular_mesh_transect(self, crs, x0, x1, y0, y1, delc, delr): # delr and delc arrays
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
        self.gi = flopy.utils.GridIntersect(self.vgrid)

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

    def create_mesh_transect(self, crs, x0, x1, y0, y1, ncol, delc): # delr is a list of column widths
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
        self.gi = flopy.utils.GridIntersect(self.vgrid)

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
                        cell = self.gi.intersect(point)["cellids"][0]
                        self.ibd[cell] = flag
                        self.obs_cells.append(cell)
                    flag += 1
                    
                
            if group == 'wel':
                for i, subgroup in enumerate(subgroups): # e.g. for 'west' in chd
                    self.cell_type.append(f'{group} - {subgroup}')
                    points = [Point(xy) for xy in spatial.xypumpbores]
                    for point in points:
                        cell = self.gi.intersect(point)["cellids"][0]
                        self.ibd[cell] = flag
                        self.wel_cells.append(cell)
                    flag += 1
                    
               
            if group == 'chd':    
                for i, subgroup in enumerate(subgroups): # e.g. for 'west' in chd    
                    self.cell_type.append(f'{group} - {subgroup}')
                    att_name = f"chd_{subgroup}_ls"
                    ls = getattr(spatial, att_name)
                    
                    cells = self.gi.intersects(ls)["cellids"]

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
                    
                    cells = self.gi.intersects(ls)["cellids"]

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
                    result = self.gi.intersect(poly)
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
            
    def plot_cell2d(self, spatial, features = None, xlim = None, ylim = None, nodes = False, labels = False):
        
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

        plt.savefig('../figures/mesh.png')
            
    def plot_feature_cells(self, spatial, xlim = None, ylim = None, plot_grid = False): # e.g xlim = [700000, 707500]
        
        fig = plt.figure(figsize=(7,5))
        spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 0.05], wspace=0.2)

        ax = fig.add_subplot(spec[0], aspect="equal") #plt.subplot(1, 1, 1, aspect="auto")
        if xlim: ax.set_xlim(xlim) 
        if ylim: ax.set_ylim(ylim) 
        ax.set_title('Special cells')
            
        pmv = flopy.plot.PlotMapView(ax = ax, modelgrid=self.vgrid)
        
        if plot_grid: 
            pmv.plot_grid(color = 'gray', lw = 0.4)
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

        plt.savefig('../figures/special_cells.png')

    def plot_surface_array(self, array, structuralmodel, plot_data = False, lithcode = None,
                           vmin = None, vmax = None, plot_grid = False, levels = None, title = None):
        
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
    
    '''def mesh_to_gdf(self):
        if self.plangrid == 'vor':
            vertices = self.vertices
            cells = self.cell2d


            #QAQC help if function isn't working
            for cell in self.cell2d:
                print("cell:", cell)
                vertex_indices = cell[1:]
                print("vertex indices:", vertex_indices)
                print("max vertex index:", max(vertex_indices))
                print("vertices length:", len(vertices))
                #coords = [vertices[int(i)] for i in vertex_indices]

            polygons = []
            for cell in cells:
                nverts = int(cell[3])
                vertex_indices = cell[4:4 + nverts]
                coords = [vertices[int(i)] for i in vertex_indices]
                polygon = Polygon(coords)
                polygons.append(polygon)

            self.gdf = gpd.GeoDataFrame(geometry=polygons)'''
    
    def plot_array(self, array, 
                   vmin = None, 
                   vmax = None, 
                   levels = None, 
                   title = None, 
                   xlim = None, ylim = None,
                   xy = None):
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot()
        if title: ax.set_title(title)
        pmv = flopy.plot.PlotMapView(modelgrid=self.vgrid)
        t = pmv.plot_array(array, vmin = vmin, vmax = vmax)
        cbar = plt.colorbar(t, shrink = 0.5)  
        #cg = pmv.contour_array(array, levels=levels, linewidths=0.8, colors="0.75")
        if xy:# Plot points
            ax.plot(xy[0], xy[1], 'o', ms = 2, color = 'red')
        if xlim: ax.set_xlim(xlim) 
        if ylim: ax.set_ylim(ylim) 