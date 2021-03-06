import taichi as ti
from vector import *
from meshtriangle  import *
from time import time
from material import *
from scene import *
from camera import Camera
import math
import random

# switch to cpu if needed
# ti.init(arch=ti.gpu )


def makeTransformations(scale, rota_angle, trans):
   scale_mat = ti.Matrix([[scale, 0, 0, 0],
                          [0, scale, 0, 0],
                          [0, 0, scale, 0],
                          [0, 0, 0, 1]])
   # 绕y轴旋转
   rota_mat = ti.Matrix([[ti.cos(rota_angle), 0, ti.sin(rota_angle), 0],
                         [0, 1, 0, 0],
                         [-ti.sin(rota_angle), 0, ti.cos(rota_angle), 0],
                         [0, 0, 0, 1]])
   trans_mat2 = ti.Matrix([[1, 0, 0, trans[0]],
                           [0, 1, 0, trans[1]],
                           [0, 0, 1, trans[2]],
                           [0, 0, 0, 1]])
   return trans_mat2 @ rota_mat @ scale_mat


@ti.data_oriented
class Render:
    def __init__(self):
       aspect_ratio = 1.0
       self.image_width = 784
       self.image_height = int(self.image_width / aspect_ratio)

       # self.film_pixels = ti.Vector.field(3, dtype=float)
       # ti.root.dense(ti.ij,
       #               (self.image_width, self.image_height)).place(self.film_pixels)
       self.samples_per_pixel = 60
       self.max_depth = 10

       # camera
       vfrom = Point([278.0, 273.0, -800.0])
       # vfrom = Point([13.0, 2.0, 3.0])
       at = Point([278.0, 273.0, 0.0])
       # at = Point([0.0, 0.0, 0.0])
       up = Vector([0.0, 1.0, 0.0])
       focus_dist = 10.0
       aperture = 0.0
       self.camera = Camera(vfrom, at, up, 40.0, aspect_ratio, aperture, focus_dist)

       red = Lambert([0.65, .05, .05])
       white = Lambert([.73, .73, .73])
       green = Lambert([.12, .45, .15])
       light = Lambert_light([15, 15, 15])
       left = MeshTriangle("./models/cornellbox/left.obj", red)
       right = MeshTriangle("./models/cornellbox/right.obj", green)
       floor = MeshTriangle("./models/cornellbox/floor.obj", white)
       light_ = MeshTriangle("./models/cornellbox/light.obj", light)


       self.base_objs=[]
       self.base_objs.append((left,0))
       self.base_objs.append((right,0))
       self.base_objs.append((floor,0))
       self.base_objs.append((light_,1))


    def init_ti(self ):
       self.film_pixels = ti.Vector.field(3, dtype=float)
       ti.root.dense(ti.ij,
                     (self.image_width, self.image_height)).place(self.film_pixels)
       self.init_film_pixels()
    @ti.func
    def ray_color(self,ray_org, ray_dir,k):

       col = ti.Vector([0.0, 0.0, 0.0])
       coefficient = ti.Vector([1.0, 1.0, 1.0])

       for i in range(self.max_depth):

          hit, p, n, front_facing, obj_index, tri_index = self.scene.hit_all(ray_org, ray_dir)

          if hit:
             isLight, emittedCol = self.scene.materials.emitted(obj_index, ray_dir, p, n, front_facing)

             if isLight:  # 光源
                # if i==0:
                if front_facing:
                   col = coefficient * emittedCol
                # print("isLight", index)
                break
             else:
                reflected, out_origin, out_direction, attenuation, mat_type = self.scene.scatter(
                   obj_index, ray_dir, p, n, front_facing, tri_index)

                if mat_type == 2:
                   ray_org, ray_dir = out_origin, out_direction.normalized()
                   coefficient *= attenuation * abs(ray_dir.dot(n))  # 衰减

                # elif front_facing:
                else:

                   pdf, ray_out_dir = self.scene.mix_sample(obj_index, ray_dir, p, n, front_facing)

                   if pdf > 0.0 and ray_out_dir.norm() > 0:

                      obj_pdf = self.scene.materials.sample_pdf(obj_index, ray_dir, p, n, front_facing, ray_out_dir)
                      ray_dir = ray_out_dir.normalized()

                      coefficient *= clamp(obj_pdf * attenuation / pdf, 0, 1)
                      ray_org = p


                   else:
                      col = ti.Vector([0.0, 0.0, 0.0])
                      break
                # else:
                #       col = ti.Vector([0.0, 0.0, 0.0])
                #       break
          else:
             col = ti.Vector([0.0, 0.0, 0.0])
             break
       return col


    def set_moddel(self,fileneme):
       self.scene = Scene()
       # self.scene.clear()
       white = Lambert([.73, .73, .73])
       spot = MeshTriangle(fileneme, white, None, "./models/spot/spot_texture.png")
       for obj,islight in self.base_objs:
          self.scene.add(obj, islight)
       self.scene.add(spot)
       self.scene.commit()

    @ti.kernel
    def init_film_pixels(self):
       for i, j in self.film_pixels:
          self.film_pixels[i, j] =ti.Vector.zero(float, 3)

    @ti.kernel
    def cal_film_val(self):
       for i, j in self.film_pixels:
          val = self.film_pixels[i, j]/self.samples_per_pixel
          self.film_pixels[i, j] =clamp(ti.sqrt(val), 0.0, 0.999)

    @ti.kernel
    def render(self,k:ti.i32):

       for i, j in self.film_pixels:
          # col = ti.Vector.zero(float, 3)

          # for k in range(self.samples_per_pixel):
             (u, v) = ((i + ti.random()) / self.image_width, (j + ti.random()) / self.image_height)
             ray_org, ray_dir = self.camera.get_ray(u, v)
             ray_dir = ray_dir.normalized()
             self.film_pixels[i, j] +=self.ray_color(ray_org, ray_dir,k)
             # if i==100 and j==100 and self.film_pixels[i, j][0]==0.0:
             #     print(ray_dir)
          #    col += self.ray_color( ray_org, ray_dir)
          # col /= self.samples_per_pixel

          # film_pixels[i, j] =  ti.sqrt(col)
          # self.film_pixels[i, j] = clamp(ti.sqrt(col), 0.0, 0.999)

    def write_img(self, filename):
       ti.imwrite(self.film_pixels.to_numpy(), filename)

    def rende_image(self,obj_filename,img_filename,i):
       self.init_ti()
       self.set_moddel(obj_filename)
       for k in range(self.samples_per_pixel):
         self.render(i)
       self.cal_film_val()
       self.write_img(img_filename)


