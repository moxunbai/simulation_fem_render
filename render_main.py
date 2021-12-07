import taichi as ti

from render import *

ti.init(arch=ti.gpu)

t2 = time()
print('starting rendering')
render= Render()
for k in range(900):
    obj_fname="./out/obj/spot_falling_"+str(k)+".obj"
    img_fname="./out/images/spot_falling_"+str(k)+".png"
    render.rende_image(obj_fname,img_fname)
    print("render image:"+img_fname)
print(time() - t2)