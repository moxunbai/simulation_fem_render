import taichi as ti
import math
import meshio
from loader.objloader import *
from util import *
from render import *

ti.init(arch=ti.gpu)

N=-1
N_tetras=-1

using_auto_diff = False
damping_toggle = ti.field(ti.i32, ())
curser = ti.Vector.field(2, ti.f64, ())
# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f64, ())
PoissonsRatio = ti.field(ti.f64, ())
LameMu = ti.field(ti.f64, ())
LameLa = ti.field(ti.f64, ())

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 100
# time-step size (for time integration)
dh = h/substepping



mesh = meshio.read(
        "./models/spot/spot_triangulated_good.mesh",
    )
obj_data =OBJ( "./models/spot/spot_triangulated_good.obj")
obj_vets=np.asarray(obj_data.vertices)
#原始obj文件表面顶点数量（比mesh文件的多）
OBJ_N=len(obj_data.vertices)
N=len(mesh.points)
N_tetras=len(mesh.cells_dict['tetra'])
# simulation components
obj_vets=ti.Vector.field(3, ti.f64, OBJ_N )
obj_normals=ti.Vector.field(3, ti.f64, OBJ_N )
#原obj文件中顶点与mesh文件顶点映射关系；因为obj经工具一系列变换成mesh后，顶点顺序变了，而原obj的面、纹理坐标、法线都是根据原始顺序定的
obj_map_mesh=ti.field(dtype=ti.i32, shape=(OBJ_N) )

obj_vets.from_numpy(np.asarray(obj_data.vertices))
obj_normals.from_numpy(np.asarray(obj_data.normals))


points=ti.Vector.field(3, ti.f64, N, needs_grad=True)
points.from_numpy(mesh.points)
v=ti.Vector.field(3, ti.f64, N, needs_grad=True)
grad = ti.Vector.field(3, ti.f64, N)
elements_Dm_inv = ti.Matrix.field(3, 3, ti.f64, N_tetras)
elements_V0 = ti.field(ti.f64, N_tetras)


# geometric components
tetras = ti.Vector.field(4, ti.i32, N_tetras)
tetras.from_numpy(mesh.cells_dict['tetra'])

#原obj模型三角形面相关数据
faces=[]
N_FACE=len(obj_data.faces)
faces_field=ti.Vector.field(3, ti.i32, N_FACE)

for  face, norms, texcoords, material in obj_data.faces:
    faces.append(face)
faces_field.from_numpy(np.asarray(faces))

@ti.kernel
def initialize():
    YoungsModulus[None] = 2e2
    PoissonsRatio[None] =0.46
    for i in range(N):
      pv = points[i]
      for j in range(OBJ_N):
        pa =obj_vets[j]
        if is_point_e(pv,pa):
            obj_map_mesh[j] =i

    #先把原本在原点附近的小牛挪到上面
    trans_matrix=makeTransformations(110,math.pi/6,[285,355,245])
    for i in range(N):
        p=points[i]
        pt=trans_matrix@ti.Vector([p[0],p[1],p[2],1])
        points[i]=ti.Vector([pt[0],pt[1],pt[2]])
    for i in range(OBJ_N):
        p=obj_vets[i]
        pt=trans_matrix@ti.Vector([p[0],p[1],p[2],1])
        obj_vets[i]=ti.Vector([pt[0],pt[1],pt[2]])
        n=obj_normals[i]
        nt=trans_matrix@ti.Vector([n[0],n[1],n[2],0])
        obj_normals[i]=ti.Vector([nt[0],nt[1],nt[2]])
    # init position and velocity
    for i in range(N):

        v[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.func
def compute_D(i):
    a = tetras[i][0]
    b = tetras[i][1]
    c = tetras[i][2]
    d = tetras[i][3]
    return ti.Matrix.cols([points[b] - points[a], points[c] - points[a], points[d] - points[a]])

@ti.kernel
def initialize_elements():
    for i in range(N_tetras):
        Dm = compute_D(i)
        elements_Dm_inv[i] = Dm.inverse()
        elements_V0[i] = ti.abs(Dm.determinant())/6

@ti.kernel
def updateLameCoeff():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    PoissonsRatio[None] = 0.499
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))

@ti.func
def compute_R_2D(F):
    R, S = ti.polar_decompose(F, ti.f64)
    return R


@ti.kernel
# @ti.func
def compute_gradient():
    # clear gradient
    for i in grad:
        grad[i] = ti.Vector([0, 0, 0])

    # gradient of elastic potential
    for i in range(N_tetras):
        Ds = compute_D(i)
        F = Ds@elements_Dm_inv[i]
        # co-rotated linear elasticity
        R = compute_R_2D(F)
        Eye = ti.Matrix.cols([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # first Piola-Kirchhoff tensor
        P = 2*LameMu[None]*(F-R) + LameLa[None]*((R.transpose())@F-Eye).trace()*R
        #assemble to gradient
        H = elements_V0[i] * P @ (elements_Dm_inv[i].transpose())
        a,b,c,d = tetras[i][0],tetras[i][1],tetras[i][2],tetras[i][3]
        gb = ti.Vector([H[0,0], H[1, 0], H[2, 0]])
        gc = ti.Vector([H[0,1], H[1, 1], H[2, 1]])
        gd = ti.Vector([H[0,2], H[1, 2], H[2, 2]])
        ga = -gb-gc-gd
        grad[a] += ga
        grad[b] += gb
        grad[c] += gc
        grad[d] += gd


@ti.kernel
# @ti.func
def update():
    # perform time integration
    for i in range(N):
        # symplectic integration
        # elastic force + gravitation force, divding mass to get the acceleration
        if using_auto_diff:
            acc = -points.grad[i]/m - ti.Vector([0.0, g,0.0])
            v[i] += dh*acc
        else:
            acc = -grad[i]/m - ti.Vector([0.0, g,0.0])
            v[i] += dh*acc
        points[i] += dh*v[i]
        # Ds = compute_D(i)
        # F = Ds @ elements_Dm_inv[i]
        # obj_normals[i] =getTransMatrix(F ,dh*v[i],obj_normals[i])

    # explicit damping (ether drag)
    for i in v:
        if damping_toggle[None]:
            v[i] *= ti.exp(-dh*5)

    # enforce boundary condition
    for i in range(N):
        if points[i][1]<=0:
            # points[i] = curser[None]
            v[i] = ti.Vector([0, 0, 0])
            points[i] = ti.Vector([points[i][0], 0.0, points[i][2]])

@ti.kernel
def update_normal():

    for i in range(OBJ_N):
        vertexNormal =ti.Vector([0.0,0.0,0.0])

        totalArea = 0.0

        for j in range(N_FACE):
            idx1=-1
            idx2=-1

            f=faces_field[j]
            if f[0]-1==i:
                idx1 = f[1]-1
                idx2 = f[2]-1
            if f[1]-1==i:
                idx1 = f[2]-1
                idx2 = f[0]-1
            if f[2]-1==i:
                idx1 = f[0]-1
                idx2 = f[1]-1
            if idx1>-1 and idx2>-1:

               e1 = points[obj_map_mesh[idx1]]-points[obj_map_mesh[i]]
               e2 = points[obj_map_mesh[idx2]]-points[obj_map_mesh[i]]
               # angle = ti.acos(e1.dot(e2))
               # if i==0:
               #   print(e1.dot(e2))
               area = e1.cross(e2).norm() * 0.5
               vertexNormal+= e1.cross(e2).normalized()*area
               totalArea+=area
        if totalArea>0.0:
            obj_normals[i] = vertexNormal/totalArea


def gen_obj(obj_filename,j):
    newVets = []
    points_arr = points.to_numpy()
    pidx_arr = obj_map_mesh.to_numpy()
    for i in range(OBJ_N):
        k = pidx_arr[i]
        newVets.append(points_arr[k])
    obj_data.vertices=newVets
    obj_data.normals=obj_normals.to_numpy()

    obj_data.write(obj_filename)

initialize()
initialize_elements()
updateLameCoeff()



t1 = time()
print('starting simulation')
for k in range(900):
   for i in range(substepping):
       compute_gradient()
       update()

   update_normal()

   gen_obj("./out/obj/spot_falling_"+str(k)+".obj",k)
print(time() - t1)
#

# t2 = time()
# print('starting rendering')
# render= Render()
# for k in range(900):
#     obj_fname="./out/obj/spot_falling_"+str(k)+".obj"
#     img_fname="./out/images/spot_falling_"+str(k)+".png"
#     render.rende_image(obj_fname,img_fname)
#     print("render image:"+img_fname)
# print(time() - t2)