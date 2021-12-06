import math
import taichi as ti

@ti.func
def round_up(value):
    # 替换内置round函数,实现保留2位小数的精确四舍五入
    a= value * 1000000

    if a-ti.cast(a,ti.i32)>=0.5:
        a+=1
    if a-ti.cast(a,ti.i32)<=-0.5:
        a-=1
    return  ti.cast(a,ti.i32)
@ti.func
def is_point_e(p1,p2):
    return (round_up(p1[0])==round_up(p2[0]) and round_up(p1[1])==round_up(p2[1]) and round_up(p1[2])==round_up(p2[2]))

@ti.func
def makeTransformations(scale,rota_angle,trans):
   scale_mat = ti.Matrix([[scale, 0, 0, 0],
                          [0, scale, 0, 0],
                          [0, 0, scale, 0],
                          [0, 0, 0, 1]])
   #绕y轴旋转
   rota_mat = ti.Matrix([[ti.cos(rota_angle), 0, ti.sin(rota_angle), 0],
                         [0, 1, 0, 0],
                         [-ti.sin(rota_angle), 0, ti.cos(rota_angle), 0],
                         [0, 0, 0, 1]])
   trans_mat = ti.Matrix([[1, 0, 0, trans[0]],
                           [0, 1, 0, trans[1]],
                           [0, 0, 1, trans[2]],
                           [0, 0, 0, 1]])
   return trans_mat@rota_mat@scale_mat

@ti.func
def getTransMatrix(f,t,v):
    m= ti.Matrix([[f[0,0], f[0,1], f[0,2], t[0]],
               [f[1,0], f[1,1], f[1,2], t[1]],
               [f[2,0], f[2,1], f[2,2], t[2]],
               [0, 0, 0, 1]])
    a=m@ti.Vector([v[0],v[1],v[2],0])
    return ti.Vector([a[0],a[1],a[2]])
    # return  f@v
    # return  (f@v).normalized()