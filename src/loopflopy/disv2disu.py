# Modified by Jim McCallum from Christian Langevin to pop cells with 0 thickness for DISU
#Hello

from collections import defaultdict
import numpy as np

class ConnectionProperty():
    def __init__(self, connected_node, this_node, ihc, cl1, cl2, hwva, angldegx):
        self.connected_node = connected_node
        self.this_node = this_node
        self.ihc = ihc
        self.cl1 = cl1
        self.cl2 = cl2
        self.hwva = hwva
        self.angldegx = angldegx


class ConnectionPropertyList():
    def __init__(self):
        self.number_of_connections = None
        self.connection_dict = {}
    
    def new_connection(self, connected_node, this_node, ihc, cl1, cl2, hwva, angldegx):
        connection = ConnectionProperty(connected_node, this_node, ihc, cl1, cl2, hwva, angldegx)
        if self.number_of_connections is None:
            connection_number = 0
            self.number_of_connections = 0
        else:
            connection_number = self.number_of_connections
        self.connection_dict[connection_number] = connection
        self.number_of_connections += 1
        return connection_number


def area_of_polygon(x, y):
    """Calculates the signed area of an arbitrary polygon given its vertices
    https://stackoverflow.com/a/4682656/ (Joe Kington)
    http://softsurfer.com/Archive/algorithm_0101/algorithm_0101.htm#2D%20Polygons
    """
    area = 0.0
    for i in range(-1, len(x) - 1):
        area += x[i] * (y[i + 1] - y[i - 1])
    return area / 2.0


def distance(x1, y1, x2, y2):
    d = (x1 - x2)**2 + (y1 - y2)**2
    d = np.sqrt(d)
    return d    


def distance_normal(x0, y0, x1, y1, x2, y2):
    d = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    d = d / distance(x1, y1, x2, y2)
    return d


def deg_anglex(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    ax = np.arctan2(dx, -dy)
    if ax < 0.:
        ax = 2.0 * np.pi + ax
    return ax * 180. / np.pi


class Disv2Disu():

    def __init__(self, vertices, cell2d, top, botm, staggered, disv_idomain = None):
        
        assert isinstance(vertices, list)
        assert isinstance(cell2d, list)
        assert isinstance(top, np.ndarray)
        assert isinstance(botm, np.ndarray)

        assert top.ndim == 1, f"top must have ndim = 2 but found {top.ndim}"
        assert botm.ndim == 2, f"botm must have ndim = 3 but found {botm.ndim}"

        self.nlay, self.ncpl = botm.shape
        self._nodes = self.nlay * self.ncpl

        assert len(cell2d) == self.ncpl, f"cell2d must have {self.ncpl} entries but found {len(cell2d)}"

        self.vertices = vertices
        self.cell2d = cell2d
        self.top = top
        self.botm = botm
        self.staggered = staggered
        
        if type(disv_idomain) == type(None):
            self.cellid = np.ones_like(botm, dtype = int)
            for j in range(self.nlay):
                self.cellid[j,:] = int(np.cumsum(self.cellid[j,:]))
                if j > 0:
                    self.cellid[j,:] += self.cellid[j-1,-1]                                             
        else:
            #had issues with the pass through cells.
            self.cellid = np.copy(disv_idomain)
            self.cellid[disv_idomain < 1] = 0
            for j in range(self.nlay):
                self.cellid[j,:] = np.cumsum(self.cellid[j,:])
                if j > 0:
                    self.cellid[j,:] += self.cellid[j-1,-1] 
            self.cellid[disv_idomain < 1] = -998 
        self.cellid -= 1 #back to python zero reference
        self.cellid = self.cellid.astype(int)
        
        self.connection_list = ConnectionPropertyList()
        self._initialize()
        self._process_neighbors()
        self._build_connectivity()
        self._modify_cell2d()

        return

    def _initialize(self):

        # unknown at this point
        self._nja = None
        self._nvert = None
        self._iac = []#np.zeros(self._nodes, dtype=int)
        self._ja = None
        self._ihc = None
        self._cl12 = None
        self._hwva = None
        self._angldegx = None
        self._vertices = None
        self._cell2d = None
        self._iverts = None

        # can be calculated
        self._top = self.top.flatten()
        top = []
        bot = []
        for k in range(self.nlay):
            for icpl in range(self.ncpl):
                if self.cellid[k,icpl] >= 0:
                    tp, bt = self.get_cell_topbot(k, icpl)
                    top.append(tp)
                    bot.append(bt)
        self._top = np.array(top, dtype=float)
        self._bot = np.array(bot, dtype=float)
        
        #self._bot = self.botm.flatten()
        self._process_vertices()
        self._fill_area()

        return

    def _process_vertices(self):
        self._nvert = len(self.vertices)
        self._xv = np.empty(self._nvert, dtype=float)
        self._yv = np.empty(self._nvert, dtype=float)
        for i, xv, yv in self.vertices:
            self._xv[i] = xv
            self._yv[i] = yv
        return

    def _process_neighbors(self):
        method = "rook"
        node_num = 0
        iverts = self.iverts
        neighbors = {i: list() for i in range(len(iverts))}
        edge_set = {i: list() for i in range(len(iverts))}
        geoms = []
        node_nums = []
        if method == "rook":
            for poly in self.iverts:
                for v in range(len(poly)):
                    geoms.append(tuple(sorted([poly[v - 1], poly[v]])))
                node_nums += [node_num] * len(poly)
                node_num += 1
        else:
            # queen neighbors
            for poly in self.iverts:
                for vert in poly:
                    geoms.append(vert)
                node_nums += [node_num] * len(poly)
                node_num += 1

        edge_nodes = defaultdict(set)
        for i, item in enumerate(geoms):
            edge_nodes[item].add(node_nums[i])

        shared_vertices = []
        for edge, nodes in edge_nodes.items():
            if len(nodes) > 1:
                shared_vertices.append(nodes)
                for n in nodes:
                    edge_set[n].append(edge)
                    neighbors[n] += list(nodes)
                    try:
                        neighbors[n].remove(n)
                    except:
                        pass

        # convert use dict to create a set that preserves insertion order
        self._neighbors = {
            i: list(dict.fromkeys(v)) for i, v in neighbors.items()
        }
        self._edge_set = edge_set


    def _fill_area(self):
        area = np.empty((self.nlay, self.ncpl), dtype=float)
        for icpl in range(self.ncpl):
            cell2drec = self.cell2d[icpl]
            nvert = cell2drec[3]
            iverts = cell2drec[4:]
            assert len(iverts) == nvert, f"cell {icpl} has nvert = {nvert} but only {len(iverts)} vertices listed in cell2d"
            a = area_of_polygon(self._xv[iverts], self._yv[iverts])
            area[:, icpl] = abs(a)
        self._area = np.array(area, dtype=float).flatten()
        self._area = self._area[self.cellid.flatten()>-1]
        
        return

    def _build_connectivity(self):
        list_of_connections = []
        self.ldum = []
        kk, iicpl = np.where(np.ones((self.nlay, self.ncpl)) == 1)
        for icell, (k, icpl) in enumerate(zip(kk, iicpl)):
            if self.cellid[k,icpl]>-1:
                connections = self.get_cell_connections(k, icpl)
                self._iac.append(len(connections) + 1)
                list_of_connections.append(connections)
                self.ldum.append(k)
        self.list_of_connections = list_of_connections
        self._iac = np.array(self._iac)

        self._nja = self._iac.sum()
        self._allocate_connection_arrays(self.nja)
        self._fill_connection_arrays()
        self._fill_vertices()
        self._fill_cell2d()

        return

    def _allocate_connection_arrays(self, nja):
        self._ja = np.zeros(nja, dtype=int)
        self._ihc = np.zeros(nja, dtype=int)
        self._cl12 = np.zeros(nja, dtype=float)
        self._hwva = np.zeros(nja, dtype=float)
        self._angldegx = np.zeros(nja, dtype=float)
        return

    def _fill_connection_arrays(self):
        ipos = 0
        for icell, cell_connections in enumerate(self.list_of_connections):
            #layer, icpl = self.get_kicpl(icell)
            self._ja[ipos] = icell
            # store layer number in ihc diagonal for flopy plotting
            self._ihc[ipos] = self.ldum[icell] + 1  #hmmmmm
            self._cl12[ipos] = icell + 1
            self._hwva[ipos] = icell + 1
            self._angldegx[ipos] = icell + 1
            ipos += 1
            cc = cell_connections.copy()
            cc.sort()
            for nn, cn in cc:
                cp = self.connection_list.connection_dict[cn]
                assert icell == cp.this_node
                self._ja[ipos] = cp.connected_node
                self._ihc[ipos] = cp.ihc
                self._cl12[ipos] = cp.cl1
                self._hwva[ipos] = cp.hwva
                self._angldegx[ipos] = cp.angldegx
                ipos += 1
        return

    def _fill_vertices(self):
        self._vertices = self.vertices
        self._nvert = len(self.vertices)
        return

    def _fill_cell2d(self):
        cell2d = []
        for k in range(self.nlay):
            for l in self.cell2d:
                icpl = l[0]
                cell2d.append([icpl + k * self.ncpl] + l[1:])
        self._cell2d = cell2d
     
    def _modify_cell2d(self):
        cell2d = []
        for k in range(self.nlay):
            for l in self.cell2d:
                icpl = l[0]
                if self.cellid[k,icpl] > -1:
                    cell2d.append([self.cellid[k,icpl]] + l[1:])
        self._cell2d = cell2d

    def get_gridprops_disu6(self):
        gridprops = {}
        self._nodes = len(self._iac)
        gridprops["nodes"] = self._nodes
        gridprops["top"] = self._top
        gridprops["bot"] = self._bot
        gridprops["area"] = self._area
        gridprops["iac"] = self._iac
        gridprops["nja"] = self._nja
        gridprops["ja"] = self._ja
        gridprops["cl12"] = self._cl12
        gridprops["ihc"] = self._ihc
        gridprops["hwva"] = self._hwva
        gridprops["angldegx"] = self._angldegx
        gridprops["nvert"] = self._nvert
        gridprops["vertices"] = self._vertices
        gridprops["cell2d"] = self._cell2d
        return gridprops

    @property
    def iac(self):
        return self._iac
        
    @property
    def nja(self):
        return self._nja
        
    def get_cell_connections(self, k, icpl):
        # return list of connections as [(nn, connection_number), ...]
        n0 = self.cellid[k, icpl]
        xc0 = self.cell2d[icpl][1]
        yc0 = self.cell2d[icpl][2]

        connected_nodes = []

        # top connection
        if k > 0:
            j = 1
            while k - j > -1:
                n1 = self.cellid[k - j, icpl]
                if n1>=0:
                    ihc = 0
                    cl1 = 0.5 * (self._top[n0] - self._bot[n0])
                    cl2 = 0.5 * (self._top[n1] - self._bot[n1])
                    hwva = self._area[n0]
                    angldegx = 0.
                    connection_number = self.connection_list.new_connection(n1, n0, ihc, cl1, cl2, hwva, angldegx)
                    connected_nodes.append((n1, connection_number))
                    break
                else:
                    j+=1

        # horizontal connections
        cell_neighbors = self._neighbors[icpl]
        cell_edges = self._edge_set[icpl]
        for icpl1, e1 in zip(cell_neighbors, cell_edges):
            k1 = k
            n1 = self.cellid[k1, icpl1]
            x1 = self._xv[e1[0]]
            y1 = self._yv[e1[0]]
            x2 = self._xv[e1[1]]
            y2 = self._yv[e1[1]]
            xc1 = self.cell2d[icpl1][1]
            yc1 = self.cell2d[icpl1][2]
            cl1 = distance_normal(xc0, yc0, x1, y1, x2, y2)
            cl2 = distance_normal(xc1, yc1, x1, y1, x2, y2)
            hwva = distance(x1, y1, x2, y2)

            # The edge variable e1 is a sorted tuple of two vertex
            # numbers.  Because the tuple is sorted, and the order
            # matters, we have to figure out if vertex order is
            # relative to n0 or n1, because angldegx must point
            # outward from the cell center.
            ivs = self.iverts[icpl]
            lenivs = len(ivs)
            ivs = ivs + [ivs[0]]
            node0_edge = False
            for i in range(lenivs):
                if (ivs[i], ivs[i + 1]) == e1:
                    node0_edge = True
                    break                    
            if node0_edge:
                angldegx = deg_anglex(x1, y1, x2, y2)
            else:
                angldegx = deg_anglex(x2, y2, x1, y1)

            if self.staggered:
                connections = self.get_connected_staggered((k, icpl), (k1, icpl1), cl1, cl2, hwva, angldegx)
                connected_nodes.extend(connections)
            else:
                if n1 >= 0:
                    ihc = 1
                    connection_number = self.connection_list.new_connection(n1, n0, ihc, cl1, cl2, hwva, angldegx)
                    connected_nodes.append((n1, connection_number))

        # bottom connections
        if k < self.nlay - 1:
            j = 1
            while k+j < self.nlay:
                n1 = self.cellid[k+j, icpl]
                if n1 >= 0:
                    ihc = 0
                    cl1 = 0.5 * (self._top[n0] - self._bot[n0])
                    cl2 = 0.5 * (self._top[n1] - self._bot[n1])
                    hwva = self._area[n1]
                    angldegx = 0.
                    connection_number = self.connection_list.new_connection(n1, n0, ihc, cl1, cl2, hwva, angldegx)
                    connected_nodes.append((n1, connection_number))
                    break
                else:
                    j+=1

        return connected_nodes

    def get_cell_topbot(self, k, icpl):
        if k == 0:
            top = self.top[icpl]
        else:
            top = self.botm[k - 1, icpl]
        bot = self.botm[k, icpl]
        return top, bot

    def get_connected_staggered(self, this_cell, connected_cell, cl1, cl2, hwva, angldegx):
        connected_cells = []
        k0, icpl0 = this_cell
        this_node = self.cellid[k0, icpl0]
        this_top, this_bot = self.get_cell_topbot(k0, icpl0)
        k1, icpl1 = connected_cell
        # look down stack for cell icpl1
        for ksearch in range(self.nlay):
            if self.cellid[ksearch,icpl1] >= 0:
                top, bot = self.get_cell_topbot(ksearch, icpl1)
                dz = min(this_top, top) - max(this_bot, bot)
                if dz > 0.1:
                    # there is overlap so add a connection
                    n1 = self.cellid[ksearch, icpl1]
                    ihc = 2
                    connection_number = self.connection_list.new_connection(n1, this_node, ihc, cl1, cl2, hwva, angldegx)
                    connected_cells.append((n1, connection_number))
        return connected_cells

    def get_nodenumber(self, k, icpl):
        return self.ncpl * k + icpl

    def get_kicpl(self, nn):
      ilay = int(nn / self.ncpl)
      icpl = int(nn - ilay * self.ncpl)
      return ilay, icpl

    @property
    def iverts(self):
        if self._iverts is None:
            self._iverts = [list(t)[4:] for t in self.cell2d]
        return self._iverts
