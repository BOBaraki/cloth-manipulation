import numpy as np
import cv2
import torch

label_file = "/home/rjangir/Pictures/kinova_dataset/" + "vertice_labels" + ".npy"

label_matrix = np.load(label_file)

print("loaded labels have shape ", label_matrix.shape)
for i in range(10):
	img_name = name = "/home/rjangir/Pictures/kinova_dataset/" + "image_" +str(i) + ".jpg"
	temp_img = cv2.imread(img_name)
	for p in range(label_matrix.shape[1]):
		cv2.circle(temp_img, (int(label_matrix[i][p][0]), int(label_matrix[i][p][1])), 10, (255, 255, 0), 5)
	cv2.imshow("out", temp_img)
	cv2.waitKey()
