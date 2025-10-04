import os
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,MultiPoint,shape
import geopandas as gpd
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import loopflopy
from loopflopy.mesh_routines import resample_linestring #, resample_poly 

def make_bbox_shp(x0, x1, y0, y1, crs, fname):
    xcoords = [x0, x1, x1, x0, x0]
    ycoords = [y0, y0, y1, y1, y0]
    bbox = Polygon(list(zip(xcoords, ycoords)))
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs = crs)
    bbox_gdf.to_file(fname)
    return bbox

def remove_duplicate_points(polygon):
    # Extract unique points using LinearRing
    linear_ring = LinearRing(polygon.exterior.coords)
    unique_coords = list(linear_ring.coords)
    # Reconstruct the Polygon without duplicates
    return Polygon(unique_coords)

    # Apply to a GeoDataFrame
    #gdf = gpd.GeoDataFrame({'geometry': [polygon]})

def create_bbox(spatial, x0, x1, y0, y1): # Create a bounding box shape file
    
    xcoords = [x0, x1, x1, x0, x0]
    ycoords = [y0, y0, y1, y1, y0]
    bbox = Polygon(list(zip(xcoords, ycoords)))
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs = spatial.epsg)
    
    if not os.path.isdir('../data/'): os.makedirs('../data/', exist_ok=True)
    if not os.path.isdir('../data/data_shp/'): os.makedirs('../data/data_shp/', exist_ok=True)
    if not os.path.isdir('../data/data_shp/bbox/'): os.makedirs('../data/data_shp/bbox/', exist_ok=True)
    bbox_gdf.to_file('../data/data_shp/bbox/bbox.shp')

def model_boundary(spatial):

    model_boundary_shp_fname = '../data/data_shp/coast/Coastline_LGATE_070.shp'
    model_boundary_gdf = gpd.read_file(model_boundary_shp_fname)
    model_boundary_gdf.to_crs(epsg=spatial.epsg, inplace=True)
    model_boundary_gdf = gpd.clip(model_boundary_gdf, bbox_gdf).reset_index(drop=True)    
    model_boundary_gdf.to_file('../data/data_shp/model_boundary/model_boundary.shp')
    model_boundary_gs = model_boundary_gdf.geometry.simplify(tolerance=1000, preserve_topology=True) # simplify 
    model_boundary_poly = resample_poly(model_boundary_gs, 2000) # resample    
    inner_boundary_poly = model_boundary_poly.buffer(-1000)
    inner_boundary_gs = gpd.GeoSeries([inner_boundary_poly])
    inner_boundary_poly = resample_poly(inner_boundary_gs, 1900) 
    
    #refinement_boundary_gs = model_boundary_gdf.buffer(5000)   
    #refinement_boundary_poly = resample_poly(refinement_boundary_gs, 3000) 
    spatial.model_boundary_gdf = model_boundary_gdf
    spatial.bbox_gdf = bbox_gdf
    spatial.model_boundary_poly = model_boundary_poly
    spatial.inner_boundary_poly = inner_boundary_poly
    spatial.x0, spatial.y0, spatial.x1, spatial.y1 = model_boundary_poly.bounds

def head_boundary(spatial):    
    model_boundary_shp_fname = '../data/data_shp/coast/Coastline_LGATE_070.shp'
    model_boundary_gdf = gpd.read_file(model_boundary_shp_fname)
    model_boundary_gdf.to_crs(epsg=spatial.epsg, inplace=True)
    model_boundary_gdf = gpd.clip(model_boundary_gdf, spatial.bbox_gdf).reset_index(drop=True)    
    model_boundary_gs = model_boundary_gdf.geometry.simplify(tolerance=1000, preserve_topology=True) # simplify 
    #model_boundary_poly = resample_poly(model_boundary_gs, 2000) # resample 

    # Convert the polygon to a LineString by accessing the boundary
    #spatial.model_boundary_gdf['linestring'] = spatial.model_boundary_gdf['geometry'].boundary
    line = model_boundary_gs[0].exterior
    coords = list(line.coords)
    print(len(coords))
    coords_to_remove = []
    for coord in coords:
        x,y = coord[0], coord[1]
        if coord[0] > 380000:
            coords_to_remove.append((x,y))

    filtered_coords = [point for point in line.coords if point not in coords_to_remove]
    filtered_coords = filtered_coords[:-1] # removing last coordinate as its on the opposite end!
    chd_west_ls = LineString(filtered_coords)
    
    from shapely.affinity import translate
    chd_west_ls = translate(chd_west_ls, xoff=200, yoff=0)
    chd_west_gdf = gpd.GeoDataFrame({'geometry': [chd_west_ls]})
    spatial.chd_west_gdf = chd_west_gdf
    spatial.chd_west_ls = chd_west_ls
    
    #gdf['geometry'] = gdf['geometry'].apply(lambda geom: LineString([point for point in geom.coords if point != (1, 1)])) # Update the geometry in the GeoDataFrame
    
    #x, y = line.xy
    #fig, ax = plt.subplots(figsize = (5,5))
    #ax.plot(x, y, '-o', ms = 7, lw = 1, color='black')
    #x, y = chd_west_ls.xy
    #ax.plot(x, y, '-o', ms = 3, lw = 1, color='red')
    
    #from shapely.affinity import translate
    #chd_west_gdf['geometry'] = new_gdf['geometry'].apply(lambda geom: translate(geom, xoff=10, yoff=0))

def obs_bores(spatial):   
    obsbore_df = pd.read_excel('../data/data_dwer/Formation picks.xls', sheet_name = 'bore_info')
    obsbore_gdf = gpd.GeoDataFrame(obsbore_df, geometry=gpd.points_from_xy(obsbore_df.Easting, obsbore_df.Northing), crs="epsg:28350")
    obsbore_gdf = gpd.clip(obsbore_gdf, spatial.inner_boundary_poly).reset_index(drop=True)
    spatial.idobsbores = list(obsbore_gdf.ID)
    spatial.xyobsbores = list(zip(obsbore_gdf.Easting, obsbore_gdf.Northing))
    spatial.nobs = len(spatial.xyobsbores)
    spatial.obsbore_gdf = obsbore_gdf
    
def pump_bores(spatial):    
    spatial.xypumpbores = [(370000, 6515000), (365700, 6525000)] # # Fake pumping bores
    spatial.npump = len(spatial.xypumpbores)
    spatial.idpumpbores = ['P1', 'P2']
    spatial.xpumpbores, spatial.ypumpbores = list(zip(*spatial.xypumpbores))
    pumpbore_gdf = gpd.GeoDataFrame(pd.DataFrame({'id': spatial.idpumpbores, 'x': spatial.xpumpbores, 'y': spatial.ypumpbores}), 
                                    geometry=gpd.points_from_xy(x=spatial.xpumpbores, y=spatial.ypumpbores))
    spatial.pumpbore_gdf = pumpbore_gdf

def faults(spatial):  
    faults_gdf = gpd.read_file('../data/data_shp/baragoon_seismic/baragoon_seismic_faults.shp')
    faults_gdf.to_crs(epsg=spatial.epsg, inplace=True)
    faults_gdf = gpd.clip(faults_gdf, spatial.model_boundary_poly).reset_index(drop=True)
    
    #from meshing_routines import remove_close_points
    #threshold = 1000
    #cleaned_coords = remove_close_points(list(streams_poly.exterior.coords), threshold) # Clean the polygon exterior
    #streams_poly = Polygon(cleaned_coords) # Create a new polygon with cleaned coordinates
   # 
   # streams_gdf = gpd.GeoDataFrame(geometry=list(streams_multipoly.geoms))

    faults_gdf.loc[int(1), "id"]  = 'Leed1'
    faults_gdf.loc[int(2), "id"] = 'Horstye'
    faults_gdf.loc[int(4), "id"] = 'Yarra1'
    faults_gdf.loc[int(10), "id"] = 'Biggestmeanest'
    faults_gdf = faults_gdf.drop(index=[0, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14])
    faults_gdf = faults_gdf.reset_index(drop = True)
    
    #### FAULTS AS LINESTRINGS
        
    r = 1000 # distance between points
    threshold_distance = 500 # Don't want fault nodes too close to boundaries.
    faults_ls, faults_coords = [], []
    faults1_coords, faults2_coords = [], []

    for i, ls in enumerate(faults_gdf.geometry): # For each fault...    
        
        # STEP 1: Resample fault to have evenly spaced nodes at distance r
        ls_resample = resample_linestring(ls, r) # Resample linestring

        # STEP 2: Removing nodes too close to inner and outer boundary so mesh doesn't go crazy refined (threshold_distance)
        nodes_to_remove = []
        for p1 in ls_resample:
            for p2 in spatial.inner_boundary_poly.exterior.coords:
                p2 = Point(p2)
                if p1.distance(p2) <= threshold_distance:
                    nodes_to_remove.append(p1)
            for p3 in spatial.model_boundary_poly.exterior.coords:
                p3 = Point(p3)
                if p1.distance(p3) <= threshold_distance:
                    nodes_to_remove.append(p1)
        if nodes_to_remove:
            print('Removing faults nodes on ', faults_gdf.loc[i, 'id'], ' because too close to boundary: ', nodes_to_remove)
        ls_new = [node for node in ls_resample if node not in nodes_to_remove]
        faults_ls.append(ls_new) # list of shapely points

        # STEP 3: Don't include short faults with only 1 node
        p = []
        for point in ls_new:
            x,y = point.x, point.y
            p.append((x,y))
        if len(p) > 1: # just making sure very short dykes with 1 point are not included
            faults_coords.append(p)
    
    faults_nodes = list(itertools.chain.from_iterable(faults_coords))
    faults_multipoint = MultiPoint(faults_nodes)
    
    spatial.faults_gdf = faults_gdf   
    spatial.faults_nodes = faults_nodes
    spatial.faults_multipoint = faults_multipoint

def streams(spatial):  
    streams_gdf = gpd.read_file('../data/data_shp/Streams.shp')
    streams_gdf.to_crs(epsg=28350, inplace=True)
    streams_gdf = gpd.clip(streams_gdf, spatial.model_boundary_poly).reset_index(drop=True)
    #streams_gdf.plot()
    from meshing_routines import resample_polys
    streams_multipoly = resample_polys(streams_gdf, 3000) # streams_multipoly = streams_gdf
    streams_poly = streams_multipoly.geoms[0] 
    
    from meshing_routines import remove_close_points
    threshold = 1000
    cleaned_coords = remove_close_points(list(streams_poly.exterior.coords), threshold) # Clean the polygon exterior
    streams_poly = Polygon(cleaned_coords) # Create a new polygon with cleaned coordinates
    
    streams_gdf = gpd.GeoDataFrame(geometry=list(streams_multipoly.geoms))

    spatial.streams_poly = streams_poly 

    
def plot_spatial(spatial):    
    
    fig, ax = plt.subplots(figsize = (7,7))
    ax.set_title('Perth Loop to Flopy')
       
    x, y = spatial.model_boundary_poly.exterior.xy
    ax.plot(x, y, '-o', ms = 2, lw = 1, color='black')
    x, y = spatial.inner_boundary_poly.exterior.xy
    ax.plot(x, y, '-o', ms = 2, lw = 0.5, color='black')

    for node in spatial.faults_nodes: 
        ax.plot(node[0], node[1], 'o', ms = 3, color = 'lightblue', zorder=2)
        
    spatial.faults_gdf.plot(ax=ax, markersize = 12, color = 'lightblue', zorder=2)
    spatial.chd_west_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
    spatial.obsbore_gdf.plot(ax=ax, markersize = 7, color = 'darkblue', zorder=2)
    spatial.pumpbore_gdf.plot(ax=ax, markersize = 12, color = 'red', zorder=2)
    
    for x, y, label in zip(spatial.obsbore_gdf.geometry.x, spatial.obsbore_gdf.geometry.y, spatial.obsbore_gdf.ID):
        ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 7, textcoords="offset points")
    for x, y, label in zip(spatial.pumpbore_gdf.geometry.x, spatial.pumpbore_gdf.geometry.y, spatial.pumpbore_gdf.id):
        ax.annotate(label, xy=(x, y), xytext=(2, 2), size = 10, textcoords="offset points")
    
    #### PLOTTING
    #ax.set_xlim([700000,spatial.x1]) 
    #ax.set_ylim([spatial.y0, 7470000]) 


### THIS TRIES TO REMOVES NODES THAT ARE TOO CLOSE!

'''streams_multipoint = MultiPoint(streams_poly.exterior.coords)

points_to_remove = []
for i, p1 in enumerate(dykes_multipoint.geoms):
    for j, p2 in enumerate(streams_multipoint.geoms):
        if p1.distance(streams_multipoint.geoms) < 1500:
            print('hello')
        else:
            points_to_remove.append(p1) 
print(len(dykes_multipoint.geoms))
new_points = [point for point in dykes_multipoint.geoms if point not in points_to_remove]
# Create a new MultiPoint without the removed point
new_dyke_multipoint = MultiPoint(new_points)

print(len(new_dyke_multipoint.geoms))'''

'''from shapely.ops import nearest_points

# Set minimum distance
min_distance = 2000
new_dyke_polys = []

# Function to ensure minimum distance between nodes
def enforce_min_distance(geom1, geom2, min_dist):
    if geom1.intersects(geom2):
        nodes1 = MultiPoint(geom1.exterior.coords)
        nodes2 = MultiPoint(geom2.exterior.coords)

        adjusted_nodes = []
        for i, point1 in  enumerate(nodes1.geoms):
            nearest_point = nearest_points(point1, nodes2)[1]
            distance = point1.distance(nearest_point)

            if distance < min_dist:
                # Move the point away
                move_vector = np.array(point1.coords[0]) - np.array(nearest_point.coords[0])
                move_vector = move_vector / np.linalg.norm(move_vector) * (min_dist - distance)
                new_point = Point(np.array(point1.coords[0]) + move_vector)
                adjusted_nodes.append(new_point)
            else:
                adjusted_nodes.append(point1)
        # Rebuild adjusted polygon
        adjusted_geom = Polygon([p.coords[0] for p in adjusted_nodes])
        return adjusted_geom
    return geom1

print('number of dykes = ', len(dykes_multipoly.geoms))

poly2 = streams_poly
for i in range(len(dykes_multipoly.geoms)):
    
    poly1 = dykes_multipoly.geoms[i] # for each dyke
    num_nodes = len(poly1.exterior.coords)
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': [poly1, poly2]})
    # Apply function to polygons
    adjusted_polygons = []
    for i, geom in enumerate(gdf.geometry):
        other_geom = gdf.loc[gdf.index != i, 'geometry'].union_all()
        adjusted_geom = enforce_min_distance(geom, other_geom, min_distance)
        adjusted_polygons.append(adjusted_geom)
    
    # Update GeoDataFrame
    #gdf['geometry'] = adjusted_polygons
    
    new_dyke_polys.append(gdf.geometry.iloc[0])

dykes_multipoly = MultiPolygon(new_dyke_polys)'''