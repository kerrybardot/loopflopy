
def define_mesh(mesh, modelmaxtri = 100000000):
    mesh.modelmaxtri = modelmaxtri
    mesh.boundmaxtri = 999999999
    mesh.angle = 32   # minimum triangle angles
    mesh.radius1 = 400 # 50, 200 radius of inner circle around pumping bores
    mesh.radius2 = 800 # 200, 600 radius of outer circle around pumping bores
    mesh.boremaxtri = 99999999



