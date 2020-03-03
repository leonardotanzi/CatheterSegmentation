from tqdm import tqdm
import os 
import cv2

path = "..\\Dataset\\i_30\\Validation\\Tool\\LabelsVOC\\"

for img in tqdm(os.listdir(path)):
	print(img)
	if img.endswith(".png"):
		image = cv2.imread(os.path.join(path, img))
		x = image.copy()

		'''
		x[:, :, 1] = x[:, :, 1] + x[:, :, 0] + x[:, :, 2]
		x[:, :, 2] = x[:, :, 1]
		x[:, :, 0] = x[:, :, 1]
		'''

		for i in range(1080):
			for j in range(1920):
				u = x[i, j, 0]
				if x[i][j][1] == 128:
					x[i, j, 0] = 1
					x[i, j, 1] = 1
					x[i, j, 2] = 1
				elif x[i][j][2] == 128:
					x[i, j, 0] = 2
					x[i, j, 1] = 2
					x[i, j, 2] = 2

		cv2.imwrite("..\\Dataset\\i_30\\Validation\\Tool\\Labels\\{}".format(img), x)