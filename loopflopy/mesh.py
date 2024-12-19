import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import flopy
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,MultiPoint,shape
from flopy.discretization import VertexGrid
from flopy.utils.triangle import Triangle as Triangle
from flopy.utils.voronoi import VoronoiGrid

class Mesh:    
    def __init__(self, plangrid, special_cells):       
        self.plangrid = plangrid
        self.special_cells = special_cells
        self.obs_cells = [] # 1
        self.wel_cells = [] # 2
        self.chd_cells = [] # 3
        
        
        #setattr(self, group, [])

#### 
    def create_bore_refinement(self, spatial):

        self.welnodes = []
        self.welnodes2 = []
        
        if self.plangrid == 'tri':
            
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
                
            spatial.bore_refinement_nodes = []
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

    def prepare_nodes_and_polygons(self, spatial, node_list, polygon_list):
        self.nodes = []
        for n in node_list: # e.g. n could be "faults_nodes"
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
        if self.plangrid == 'tri':
        
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
            self.ncpl = len(self.cell2d)
            self.vgrid = flopy.discretization.VertexGrid(vertices=self.vertices, cell2d=self.cell2d, ncpl = self.ncpl, nlay = 1)
            
        if self.plangrid == 'vor':
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
            self.vertices = vor.get_disv_gridprops()['vertices']
            self.cell2d = vor.get_disv_gridprops()['cell2d']
        
            self.xcyc = []
            for cell in self.cell2d:
                self.xcyc.append((cell[1],cell[2]))
    
    def locate_special_cells(self, spatial):
        self.gi = flopy.utils.GridIntersect(self.vgrid)
        self.ibd = np.zeros(self.ncpl, dtype=int) # empty array top shade important cells
        for group in self.special_cells:
            
            subgroups = self.special_cells[group]
            print('Group = ', group, subgroups)

            #for attribute, value in flowmodel.data.__dict__.items(): print(attribute)
            #for key, value in d.items():
            #print(f"{key}: {value}")

            for i, subgroup in enumerate(subgroups): # e.g. for 'west' in chd
                if group == 'obs':
                    points = [Point(xy) for xy in spatial.xyobsbores]
                    for point in points:
                        cell = self.gi.intersect(point)["cellids"][0]
                        self.ibd[cell] = 1
                        self.obs_cells.append(cell)
                if group == 'wel':
                    points = [Point(xy) for xy in spatial.xypumpbores]
                    for point in points:
                        cell = self.gi.intersect(point)["cellids"][0]
                        self.ibd[cell] = 2
                        self.wel_cells.append(cell)
                        chd_west_cells = []
                if group == 'chd':         
                    self.chd_cells = self.gi.intersects(spatial.chd_west_ls)["cellids"]
                    for cell in self.chd_cells:
                        self.ibd[cell] = 3
            
            #stream_cells = []
            #for i in mesh.xcyc:
            #    point = Point((i[0], i[1]))
            #    cell = mesh.gi.intersect(point)["cellids"]
            #    if spatial.streams_poly.contains(point):
            #        mesh.ibd[cell[0]] = 3
            #        stream_cells.append(cell[0])
            
    def plot_feature_cells(self, spatial, xlim = None, ylim = None): # e.g xlim = [700000, 707500]
    
        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(1, 1, 1, aspect="equal")
        pmv = flopy.plot.PlotMapView(modelgrid=self.vgrid)
        p = pmv.plot_array(self.ibd, alpha = 0.6)
        self.tri.plot(ax=ax, edgecolor='black', lw = 0.1)
             
        ax.set_xlim(xlim) 
        ax.set_ylim(ylim) 
           
        for group in self.special_cells:
            
            if group == 'obs':
                spatial.obsbore_gdf.plot(ax=ax, markersize = 7, color = 'darkblue', zorder=2)
                for cell in self.obs_cells:
                    ax.plot(self.cell2d[cell][1], self.cell2d[cell][2], "o", color = 'black', ms = 1)
            
            if group == 'wel':
                spatial.pumpbore_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
                for cell in self.wel_cells:
                    ax.plot(self.cell2d[cell][1], self.cell2d[cell][2], "o", color = 'blue', ms = 2)

            if group == 'chd':
                for cell in self.chd_cells:
                    ax.plot(self.cell2d[cell][1], self.cell2d[cell][2], "o", color = 'red', ms = 1)

            if group == 'zone':
                for subgroup in self.special_cells['zone']: # subgroup must be a poly
                    x, y = spatial.subgroup.exterior.xy
                    ax.plot(x, y, '-o', ms = 2, lw = 0.5, color='blue') 

    def plot_cell2d(self, spatial, features = None, xlim = None, ylim = None):
        
        fig = plt.figure(figsize=(7,7))
        ax = plt.subplot(1, 1, 1, aspect='equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title('Number cells in plan (ncpl): ' + str(len(self.cell2d)))
    
        if self.plangrid == 'car': self.sg.plot(color = 'gray', lw = 0.4) 
        if self.plangrid == 'tri': self.tri.plot(edgecolor='gray', lw = 0.4)
        if self.plangrid == 'vor': self.vor.plot(edgecolor='black', lw = 0.4)
            
        for i in self.xcyc: 
                ax.plot(i[0], i[1], 'o', color = 'green', ms = 0.5)    
        for i in self.nodes: 
            ax.plot(i[0], i[1], 'o', ms = 2, color = 'black')
        x, y = spatial.model_boundary_poly.exterior.xy
        ax.plot(x, y, '-o', ms = 2, lw = 1, color='black')
        x, y = spatial.inner_boundary_poly.exterior.xy
        ax.plot(x, y, '-o', ms = 2, lw = 0.5, color='black')
            
        if 'obs' in features:
            spatial.obsbore_gdf.plot(ax=ax, markersize = 7, color = 'darkblue', zorder=2)
            for x, y, label in zip(spatial.obsbore_gdf.geometry.x, spatial.obsbore_gdf.geometry.y, spatial.obsbore_gdf.ID):
                ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 8, textcoords="offset points")

        if 'wel' in features:
            spatial.pumpbore_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
            for i in range(spatial.npump):
                ax.plot(spatial.xypumpbores[i], ms = 2, color = 'black')   
            for x, y, label in zip(spatial.pumpbore_gdf.geometry.x, spatial.pumpbore_gdf.geometry.y, spatial.pumpbore_gdf.id):
                ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 8, textcoords="offset points")
        
        if 'fault' in features:
            spatial.faults_gdf.plot(ax=ax, color = 'red', zorder=2)










            

        

            




