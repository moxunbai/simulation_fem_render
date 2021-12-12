import taichi as ti

from render import *

# ti.init(arch=ti.gpu)


print('starting rendering')
render = Render()
for k in range(900):
    t1 = time()
    ti.reset()
    ti.init(arch=ti.cuda,random_seed=int( time()))

    obj_fname="./out/obj/spot_falling_"+str(k)+".obj"
    img_fname="./out/images/spot_falling_"+str(k)+".png"
    render.rende_image(obj_fname,img_fname,k)
    print("render image:"+img_fname)
    print(time() - t1)


