import pandas as pd
import numpy as np
from shapely.geometry import LineString,Point,Polygon,MultiPolygon,shape
import loopflopy.utils as utils
import pickle
from scipy.interpolate import griddata

class Data:
    def __init__(self):

            self.data_label = "DataBaseClass"

    def process_rch(self, geomodel, mesh):

        
        rch = 0.0001/365
        
        cells_top = []

        for icpl in range(mesh.ncpl):
            lay = 0
            while lay <= geomodel.nlay:
                
                cell_disv = icpl + lay*mesh.ncpl
                cell_disu = geomodel.cellid_disu.flatten()[cell_disv]

                if cell_disu == -1: # if cell is pinched out keep searching...
                    lay += 1

                else: # Bingo! found top cell in the pillar!
                    cells_top.append(cell_disu)
                    break 

        rec = []
        for icpl in cells_top:
            rec.append((icpl, rch))      
        self.rch_rec = {}      
        self.rch_rec[0] = rec 

    def process_wel(self, geomodel, mesh, spatial, wel_q, wel_qlay):
                  # geo layer pumping from
        
        ## Assume screening pumping well across entire geological layer, ## Find top and bottom of screen 
        self.wel_screens = []
        self.spd_wel = []
        for n in range(spatial.npump):
            icpl = mesh.wel_cells[n]
            if wel_qlay == 0:
                wel_top = geomodel.top[icpl]  
            else:   
                wel_top = geomodel.botm[(wel_qlay[n])* geomodel.nls-1, icpl]
            wel_bot = geomodel.botm[(wel_qlay[n] + 1) * geomodel.nls-1, icpl]   
            self.wel_screens.append((wel_top, wel_bot))
                   
            if geomodel.vertgrid == 'vox':
                nwell_cells = int((wel_top - wel_bot)/geomodel.dz)
                for lay in range(int((geomodel.top_geo[icpl]-wel_top)/geomodel.dz), int((geomodel.top_geo[icpl]-wel_top)/geomodel.dz) + nwell_cells):   
                    cell_disv = icpl + lay*mesh.ncpl
                    cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                    self.spd_wel.append([cell_disu, wel_q[n]/nwell_cells])
    
            if geomodel.vertgrid == 'con':        
                nwell_cells = geomodel.nls # For this research, assume pumping across entire geological layer
                for wel_lay in range(wel_qlay[n] * geomodel.nls, (wel_qlay[n] + 1) * geomodel.nls): # P.geo_pl = geological pumped layer                    
                    cell_disv = icpl + wel_lay*mesh.ncpl
                    cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                    self.spd_wel.append([cell_disu, wel_q[n]/nwell_cells])
                    
            if geomodel.vertgrid == 'con2':       
                lay = 0
                well_layers = []
                nwell_cells = 0
                
                while geomodel.botm[lay, icpl] >= wel_top-0.1: # above top of screen
                    lay += 1
                while geomodel.botm[lay, icpl] > wel_bot: # above bottom of screen
                    if geomodel.idomain[lay, icpl] != -1: # skips pinched out cells
                        nwell_cells += 1
                        well_layers.append(lay)
                    lay += 1
                
                for lay in well_layers:
                    cell_disv = icpl + lay*mesh.ncpl
                    cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                    self.spd_wel.append([cell_disu, wel_q[n]/nwell_cells])      
    
        print('Well screens ', self.wel_screens)

    def process_ic(self):
        self.strt = -10. #geomodel.top_geo - 1. # start with a watertable of 1m everywhere #wt.reshape(1,len(wt))

    def process_chd(self, spatial, geomodel, mesh):

        chfunc = lambda x,z: 0.001 * (x - 360000)- (z * 0.02)-20
        self.chd_rec = []
        
        # CHD WEST
        for icpl in mesh.chd_west_cells:
            print(mesh.chd_west_cells)
            x,y = mesh.xcyc[icpl][0], mesh.xcyc[icpl][1]
            for lay in range(geomodel.nlay):
                zc = geomodel.zc[lay, icpl]
                zb = geomodel.botm[lay,icpl]
                if zb < chfunc(x,zc):                           
                    cell_disv = icpl + lay*(mesh.ncpl)
                    cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                    if cell_disu != -1:
                        self.chd_rec.append([cell_disu, chfunc(x,zc)])         

        # CHD EAST
        for icpl in mesh.chd_east_cells:
            print(mesh.chd_east_cells)
            x,y = mesh.xcyc[icpl][0], mesh.xcyc[icpl][1]
            for lay in range(geomodel.nlay):
                zc = geomodel.zc[lay, icpl]
                zb = geomodel.botm[lay,icpl]
                if zb < chfunc(x,zc):                           
                    cell_disv = icpl + lay*(mesh.ncpl)
                    cell_disu = geomodel.cellid_disu.flatten()[cell_disv]
                    if cell_disu != -1:
                        self.chd_rec.append([cell_disu, chfunc(x,zc)])     
    


