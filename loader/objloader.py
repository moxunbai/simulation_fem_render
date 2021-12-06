import numpy as np
 
'''
the original is here https://www.pygame.org/wiki/OBJFileLoader
@2018-1-2 author chj
change for easy use
'''

#
# def MTL(fdir, filename):
#     contents = {}
#     mtl = None
#     for line in open(fdir + filename, "r"):
#         if line.startswith('#'): continue
#         values = line.split()
#         if not values: continue
#         if values[0] == 'newmtl':
#             mtl = contents[values[1]] = {}
#         elif mtl is None:
#             raise ValueError("mtl file doesn't start with newmtl stmt")
#         elif values[0] == 'map_Kd':
#             # load the texture referred to by this declaration
#             mtl[values[0]] = values[1]
#             surf = pygame.image.load(fdir + mtl['map_Kd'])
#             image = pygame.image.tostring(surf, 'RGBA', 1)
#             ix, iy = surf.get_rect().size
#             texid = mtl['texture_Kd'] = glGenTextures(1)
#             glBindTexture(GL_TEXTURE_2D, texid)
#             glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
#                             GL_LINEAR)
#             glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
#                             GL_LINEAR)
#             glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
#                          GL_UNSIGNED_BYTE, image)
#         else:
#             # mtl[values[0]] = map(float, values[1:])
#
#             mtl[values[0]] = [float(x) for x in values[1:4]]
#     return contents


class OBJ:
    def __init__(self,  filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self.mtl = None

        # self.face = ti.Vector.field(3, dtype=ti.f32)
        # self.direction = ti.Vector.field(3, dtype=ti.f32)

        material = None
        for line in open( filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                # v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                # v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                v = [float(x) for x in values[1:3]]

                self.texcoords.append(v)
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                # print(values[1])
                # self.mtl = MTL(fdir,values[1])
                self.mtl = [ values[1]]
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    # print("w len:",len(w))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

    def write(self,tofilename):
        with open(tofilename, 'w') as file_object:
            nv=len(self.vertices)
            nf=len(self.faces)
            if nv>0:
              file_object.write("# Object spot_triangulated_good.obj\n")
              file_object.write("# Vertices:"+str(nv)+"\n")
              file_object.write("# Faces:"+str(nf)+"\n")
              for i in range(nv):

                  if self.normals is not None and len(self.normals)>0:
                      nor=self.normals[i]
                      file_object.write("vn " + str(nor[0]) +" "+str(nor[1]) +" "+str(nor[2])+ "\n")
                  if self.texcoords is not None and len(self.texcoords) > 0:
                      tex = self.texcoords[i]
                      file_object.write("vt " + str(tex[0]) + " " + str(tex[1]) + "\n")
                  vet=self.vertices[i]
                  file_object.write("v " + str(vet[0]) + " " + str(vet[1]) + " " + str(vet[2]) + "\n")

              file_object.write("\n")
              for i in range(nf):
                  (face, norms, texcoords, material)=self.faces[i]
                  txt=[str(face[0]),str(face[1]),str(face[2])]

                  if  len(texcoords)>=2:
                      txt=[txt[0]+"/"+str(texcoords[0]),txt[1]+"/"+str(texcoords[1]),txt[2]+"/"+str(texcoords[2])]

                  if  len(norms)==3:
                      txt=[txt[0]+"/"+str(norms[0]),txt[1]+"/"+str(norms[1]),txt[2]+"/"+str(norms[2])]

                  file_object.write("f " + " ".join(txt) + "\n")
              file_object.write("# End of File \n")



    def create_bbox(self):
        # self.vertices is not None
        ps = np.array(self.vertices)
        vmin = ps.min(axis=0)
        vmax = ps.max(axis=0)

        self.bbox_center = (vmax + vmin) / 2
        self.bbox_half_r = np.max(vmax - vmin) / 2
