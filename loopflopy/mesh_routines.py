import matplotlib.pyplot as plt
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import numpy as np
import geopandas as gpd
import math
import flopy
from collections import OrderedDict

def resample_gs(gs, distance): # gdf contains a single polygon
    exterior_coords = list(gs.exterior.coords)
    exterior_line = LineString(exterior_coords)
    resampled_line = []
    current_distance = 0
    while current_distance <= exterior_line.length:
        resampled_line.append(exterior_line.interpolate(current_distance))
        current_distance += distance
    coords = []
    for point in resampled_line:
        x,y = point.x, point.y
        coords.append((x,y))
    poly_resampled = Polygon(coords) # Keep polygons as Shapely polygons
    return(poly_resampled) # Returns a Shapely polygon

# Function to remove duplicate points
def remove_duplicate_points(polygon):
    from shapely.geometry import Polygon, LinearRing
    # Extract unique points using LinearRing
    linear_ring = LinearRing(polygon.exterior.coords)
    unique_coords = list(linear_ring.coords)
    # Reconstruct the Polygon without duplicates
    return Polygon(unique_coords)

    # Apply to a GeoDataFrame
    #spatial.model_boundary_gdf['geometry'] = spatial.model_boundary_gdf['geometry'].apply(remove_duplicate_points)

def resample_gdf_poly(gdf, distance): # gdf contains a single polygon
    poly = gdf.geometry[0]    
    exterior_coords = list(poly.exterior.coords)    
    exterior_line = LineString(exterior_coords)
    resampled_line = []
    current_distance = 0
    while current_distance <= exterior_line.length - distance:
        resampled_line.append(exterior_line.interpolate(current_distance))
        current_distance += distance
    coords = []
    for point in resampled_line:
        x,y = point.x, point.y
        coords.append((x,y))
    poly_resampled = Polygon(coords) # Keep polygons as Shapely polygons
    return(poly_resampled) # Returns a Shapely polygon

def resample_shapely_poly(poly, distance): # gdf contains a single polygon 
    exterior_coords = list(poly.exterior.coords)    
    exterior_line = LineString(exterior_coords)
    resampled_line = []
    current_distance = 0
    while current_distance <= exterior_line.length - distance:
        resampled_line.append(exterior_line.interpolate(current_distance))
        current_distance += distance
    coords = []
    for point in resampled_line:
        x,y = point.x, point.y
        coords.append((x,y))
    poly_resampled = Polygon(coords) # Keep polygons as Shapely polygons
    return(poly_resampled) # Returns a Shapely polygon

def resample_polys(gdf, distance):
    multipoly = []
    for n in range(gdf.shape[0]):
        poly = gdf.geometry[n]    
        exterior_coords = list(poly.exterior.coords)
        exterior_line = LineString(exterior_coords)
        resampled_line = []
        current_distance = 0
        while current_distance <= exterior_line.length:
            resampled_line.append(exterior_line.interpolate(current_distance))
            current_distance += distance
        coords = []
        for point in resampled_line:
            x,y = point.x, point.y
            coords.append((x,y))
        poly_resampled = Polygon(coords) # Keep polygons as Shapely polygons
        multipoly.append(poly_resampled)
    return(MultiPolygon(multipoly)) # Returns a Shapely multipolygon

def resample_linestring(linestring, distance):
    total_length = linestring.length
    num_points = int(total_length / distance)
    points = [linestring.interpolate(distance * i) for i in range(num_points + 1)]
    return points

def get_ls_from_gdf(gdf):
    points = []
    for line in gdf['geometry']:
        x, y = line.xy
        points.extend(list(zip(x, y)))
    ls = LineString(points)
    return(ls)

#def get_xy_from_gdf(gdf):
#    points = []
#    for line in gdf['geometry']:
#        x, y = line.xy
#        points.extend(list(zip(x, y)))
#    x,y = zip(*points)
#    return(x,y)

#Define a function that returns a list of X,Y
def extract_coord_from_shape(gdf):
    coordinates = []
    for geometry in gdf.geometry:
        if geometry.geom_type == 'Polygon': # For polygons, extract X and Y coordinates
            coords = geometry.exterior.coords
            for x, y in coords:
                coordinates.append([x,y])
        elif geometry.geom_type == 'LineString': # For linestrings, extract X and Y coordinates
            coords = geometry.coords
            for x, y in coords:
                coordinates.append([x,y])    
    return coordinates

def remove_close_points(coords, threshold): # Function to remove close points
    filtered_coords = [coords[0]]  # Start with the first coordinate
    
    for coord in coords[1:]:
        # Calculate distance from the last kept point
        distance = Point(filtered_coords[-1]).distance(Point(coord))
        if distance >= threshold:
            filtered_coords.append(coord)
    
    # Ensure the polygon remains closed (first == last point)
    if filtered_coords[0] != filtered_coords[-1]:
        filtered_coords.append(filtered_coords[0])
    
    return filtered_coords


# Preparing meshes for boundaries, bores and fault. 
def prepboremesh(spatial, mesh):
    
    theta = np.linspace(0, 2 * np.pi, 11)

    pump_bores_inner, pump_bores_outer = [], []
    obs_bores_inner, obs_bores_outer = [], []
    
    if mesh.plangrid == 'tri':
        
        def vertices_equtri1(X, Y, l): # l is distance from centre of triangle to vertex
            x1 = X - l*3**0.5/2 
            x2 = X + l*3**0.5/2 
            x3 = X
            y1 = Y - l/2
            y2 = Y - l/2
            y3 = Y + l
            return(x1, x2, x3, y1, y2, y3)
        
        def vertices_equtri2(X, Y, l): # l is distance from centre of triangle to vertex
            x1 = X 
            x2 = X + l*3**0.5
            x3 = X - l*3**0.5
            y1 = Y - 2*l
            y2 = Y + l
            y3 = Y + l
            return(x1, x2, x3, y1, y2, y3)

        for i in spatial.xypumpbores:   
            X, Y = i[0], i[1] # coord of pumping bore
                        
            x1, x2, x3, y1, y2, y3 = vertices_equtri1(X, Y, mesh.radius1) #/2
            vertices_inner = ((x1, y1), (x2, y2), (x3, y3))
            x1, x2, x3, y1, y2, y3 = vertices_equtri2(X, Y, mesh.radius1) #/2
            vertices_outer = ((x1, y1), (x2, y2), (x3, y3))
            
            pump_bores_inner.append(vertices_inner)
            pump_bores_outer.append(vertices_outer)
                    
        obs_tri_vertices = []
        for i in spatial.xyobsbores:   
            X, Y = i[0], i[1] # coord of pumping bore
                        
            x1, x2, x3, y1, y2, y3 = vertices_equtri1(X, Y, mesh.obs_ref) #/2
            vertices = ((x1, y1), (x2, y2), (x3, y3))   
            obs_tri_vertices.append(vertices)

        return(pump_bores_inner, pump_bores_outer, obs_tri_vertices)#, obs_bores_inner, obs_bores_outer)
    
    if mesh.plangrid == 'vor':
        for i in P.xypumpbores:   
            X = i[0] + P.radius1 * np.cos(theta)
            Y = i[1] + P.radius1 * np.sin(theta)    
            vertices_inner = [(x_val, y_val) for x_val, y_val in zip(X, Y)]
            X = i[0] + P.radius2 * np.cos(theta)
            Y = i[1] + P.radius2 * np.sin(theta)    
            vertices_outer = [(x_val, y_val) for x_val, y_val in zip(X, Y)]
            pump_bores_inner.append(vertices_inner)
            pump_bores_outer.append(vertices_outer)
            
        #for i in P.xyobsbores:   
        #    X = i[0] + P.radius1 * np.cos(theta)
        #    Y = i[1] + P.radius1 * np.sin(theta)    
        #    vertices_inner = [(x_val, y_val) for x_val, y_val in zip(X, Y)]
        #    X = i[0] + P.radius2 * np.cos(theta)
        #    Y = i[1] + P.radius2 * np.sin(theta)    
        #    vertices_outer = [(x_val, y_val) for x_val, y_val in zip(X, Y)]
        #    obs_bores_inner.append(vertices_inner)
        #    obs_bores_outer.append(vertices_outer)

        return(pump_bores_inner, pump_bores_outer) #, obs_bores_inner, obs_bores_outer)


def prepare_fault_nodes_voronoi(P, shpfilepath, model_boundary, inner_boundary):
    # Import fault and turn into a linestring
    #gdf = gpd.read_file('../shp/badaminna_fault.shp') 
    
    gdf = gpd.read_file(shpfilepath) 
    fault = gpd.clip(gdf, model_boundary) # fault is a gdf
    df = fault.get_coordinates()
    fault_points = list(zip(list(df.x), list(df.y)))
    fault_linestring = LineString(fault_points)

    # Settings to make point cloud
    L = P.fault_buffer
    Lfault = fault.length
    r = 2*L/3 # distance between points

    # Fault point cloud
    offsets = [-1.5*r, -0.5*r, 0.5*r, 1.5*r]
    fault_offset_lines = []
    for offset in offsets:
        ls = fault_linestring.parallel_offset(offset) # linestring.parallel_offset
        ls_resample = resample_linestring(ls, r)
        p = []
        for point in ls_resample:
            if inner_boundary.contains(point):
                x,y = point.x, point.y
                p.append((x,y))
        offset_ls = LineString(p)
        coords = list(offset_ls.coords)
        fault_offset_lines.append(coords)

    fault_refinement_nodes = [tup for line in fault_offset_lines for tup in line]
    
    return(fault_refinement_nodes)
    
# PREPARING NODES AND POLYGONS AND THEN CALLING MESHING FUNCTION

def createcell2d(P, grid, fault = False):

    if grid == 'car':
        delr = P.delx * np.ones(P.ncol, dtype=float)
        delc = P.dely * np.ones(P.nrow, dtype=float)
        top  = np.ones((P.nrow, P.ncol), dtype=float)
        botm = np.zeros((1, P.nrow, P.ncol), dtype=float)
        sg = flopy.discretization.StructuredGrid(delr=delr, delc=delc, top=top, botm=botm)
        xycenters = sg.xycenters
        
        cell2d = []
        xcyc = [] # added 
        for n in range(P.nrow*P.ncol):
            l,r,c = sg.get_lrc(n)[0]
            xc = xycenters[0][c]
            yc = xycenters[1][r]
            iv1 = c + r * (P.ncol + 1)  # upper left
            iv2 = iv1 + 1
            iv3 = iv2 + P.ncol + 1
            iv4 = iv3 - 1
            cell2d.append([n, xc, yc, 5, iv1, iv2, iv3, iv4, iv1])
            xcyc.append((xc, yc))
        
        vertices = []
        xa = np.arange(P.x0, P.x1 + P.delx, P.delx)      
        ya = np.arange(P.y1, P.y0 - P.dely/2, -P.dely)

        n = 0
        for j in ya:
            for i in xa:
                vertices.append([n, i, j])
                n+=1
                
        return(cell2d, xcyc, vertices, sg)
    
    if grid == 'tri': 
        #boresinner, boresouter, obsinner, obsouter = prepboremesh(P, grid = grid)
        boresinner, boresouter, obs_tri_vertices = prepboremesh(P, grid = grid)
        modelextpoly, modelintpoly = prepboundarymesh(P, grid = grid)
            
        nodes = []
        for bore in boresinner: 
            for n in bore: nodes.append(n)
        for bore in boresouter: 
            for n in bore: nodes.append(n)
        for bore in obs_tri_vertices: 
            for n in bore: nodes.append(n)
        #for bore in obsouter: 
        #    for n in bore: nodes.append(n)
        if fault == True:
            faultpoints = prepfaultmesh(P, grid = grid)
            for point in faultpoints:
                nodes.append(point)
        if fault == False:
            if 'P.fault_poly' in locals():
                del P.fault_poly
        
        nodes = np.array(nodes)
        
        polygons = []
        polygons.append((modelextpoly, (P.x0 + 10, P.y0 + 10), P.boundmaxtri)) # Inside boundary frame
        polygons.append((modelintpoly, (P.x0 + P.w + 10, P.y0 + P.w + 10), P.modelmaxtri)) # Bulk of model!       
        cell2d, xcyc, vertices, gridobject = tri_meshing(P, polygons, nodes)
        
        return(cell2d, xcyc, vertices, gridobject, nodes)
        
    if grid == 'vor': 
        #pumpinner, pumpouter, obsinner, obsouter = prepboremesh(P, grid = grid)
        pumpinner, pumpouter = prepboremesh(P, grid = grid)
        modelextpoly, modelintpoly = prepboundarymesh(P, grid = grid)
        
        nodes = []
        #for point in modelextpoly: # Added back 29/4
        #    nodes.append(point) # Added back 29/4
        #for point in modelintpoly: # Added back 29/4
        #    nodes.append(point) # Added back 29/4
        for point in P.xypumpbores:
            nodes.append(point)
        for point in P.xyobsbores:
            nodes.append(point)
        if fault == True:
            faultpoints = prepfaultmesh(P, grid = grid)
            for point in faultpoints:
                nodes.append(point)
        if fault == False:
            if 'P.fault_poly' in locals():
                del P.fault_poly
        #import numpy as np        
        nodes = np.array(nodes)
        
        polygons = []
        polygons.append((modelextpoly, (P.x0 + 10, P.y0 + 10), P.boundmaxtri)) # Inside boundary frame
        polygons.append((modelintpoly, (P.x0 + P.w + 10, P.y0 + P.w + 10), P.modelmaxtri)) # Bulk of model!     
        
        for i in range(P.npump): # Append pumping bore zone polygons
            polygons.append((pumpinner[i], P.xypumpbores[i], P.boremaxtri))
            polygons.append((pumpouter[i],0, 0)) # 0, 0 means don't refine inside polygon
            
        #for i in range(P.nobs): # Append pumping bore zone polygons
            #polygons.append((obsinner[i], P.xyobsbores[i], P.boremaxtri))
            #polygons.append((obsouter[i],0, 0)) # 0, 0 means don't refine inside polygon
        
        cell2d, xcyc, vertices, gridobject = vor_meshing(P, polygons, nodes)
    
        return(cell2d, xcyc, vertices, gridobject, nodes)


    


    

