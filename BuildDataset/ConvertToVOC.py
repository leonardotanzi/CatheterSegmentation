from tqdm import tqdm
import os 
import cv2

path = "C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\New\\LabelsVOC\\"
out_path = "C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\New\\Labels\\"


i = 0
for img in tqdm(os.listdir(path)):
	print(img)
	if img.endswith(".png"):
		image = cv2.imread(os.path.join(path, img))
		if i == 0:
			h = image.shape[0]
			w = image.shape[1]
			print(h)
			print(w)
		x = image.copy()

		'''
		x[:, :, 1] = x[:, :, 1] + x[:, :, 0] + x[:, :, 2]
		x[:, :, 2] = x[:, :, 1]
		x[:, :, 0] = x[:, :, 1]
		'''

		for i in range(h):
			for j in range(w):
				u = x[i, j, 0]
				if x[i][j][1] == 128:
					x[i, j, 0] = 1
					x[i, j, 1] = 1
					x[i, j, 2] = 1
				elif x[i][j][2] == 128:
					x[i, j, 0] = 2
					x[i, j, 1] = 2
					x[i, j, 2] = 2

		cv2.imwrite(out_path + "{}".format(img), x)
		i += 1