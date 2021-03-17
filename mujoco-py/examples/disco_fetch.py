#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder, MaterialModder
import os
from gym import utils
from gym.envs.robotics import gen3_env

MODEL_XML_PATH = "/home/rjangir/workSpace/gen3-mujoco/gym/gym/envs/robotics/assets/gen3/sideways_fold.xml"
model = load_model_from_path(MODEL_XML_PATH)
#model = load_model_from_path("xmls/fetch/main.xml")
sim = MjSim(model)

viewer = MjViewer(sim)
modder = TextureModder(sim)
#matModder = MaterialModder(sim) #specularity, shininess, reflectance

t = 0

while True:
	#print("body names", sim.model.name_skinadr, sim.model.skin_matid[0])
	skin_mat = sim.model.skin_matid[0]
	#print("skin mat texture", sim.model.mat_texid[skin_mat], sim.model.tex_type[sim.model.mat_texid[skin_mat]])
	for name in sim.model.geom_names:
		modder.whiten_materials()
		modder.set_checker(name, (255, 0, 0), (0, 0, 0))
		modder.rand_all(name)
	modder.set_checker('skin', (255, 0, 0), (0, 0, 0))
	modder.rand_all('skin')
	viewer.render()
	t += 1
	if t > 100 and os.getenv('TESTING') is not None:
		break

