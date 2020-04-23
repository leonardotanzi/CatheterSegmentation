from typing import List, Tuple
import cv2
import numpy as np
from keras_segmentation.predict import predict, model_from_checkpoint_path
from scipy.ndimage import gaussian_filter
import argparse

class_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]

scale_percent = 60  # percent of original size


def convert_np_to_mat(img):
	seg_img = np.zeros((img.shape[0], img.shape[1], 3))
	colors = class_colors

	for c in range(3):  # with 3 classes

		seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype('float')
		seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype('float')
		seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype('float')

	seg_img = cv2.resize(seg_img, (960, 540))

	return seg_img


def build_line(p1, p2):
	a = (p1[1] - p2[1])
	b = (p2[0] - p1[0])
	c = (p1[0]*p2[1] - p2[0]*p1[1])
	return a, b, -c


def intersection(line1, line2):
	delta = line1[0] * line2[1] - line1[1] * line2[0]
	delta_x = line1[2] * line2[1] - line1[1] * line2[2]
	delta_y = line1[0] * line2[2] - line1[2] * line2[0]
	if delta != 0:
		x = delta_x / delta
		y = delta_y / delta
		return int(x), int(y)
	else:
		return 0, 0


if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model", required=True, help="Select the model to use "
														  "(0 for MobileNet, 1 for VGG, 2 for ResNet, 3 for Unet")
	args = vars(ap.parse_args())
	run_model = int(args["model"])
	models = ["MobileNet", "VGG", "ResNet", "U-Net"]
	print("Running the {} model.".format(models[run_model]))

	videoFilePath = "..\\RealTime Video\\CV_2_cropped.mp4"

	if run_model == 0:
		model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_mobilenet_unet_tool")
	elif run_model == 1:
		model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_vgg_unet_tool")
	elif run_model == 2:
		model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_resnet_unet_tool")
	elif run_model == 3:
		model = model_from_checkpoint_path("..\\Checkpoints\\NewTool\\new_unet_tool")

	cap = cv2.VideoCapture(videoFilePath)
	i = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		dim = (960, 540)

		frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
		angle = 0.0

		if (i % 10) == 0:
			out = model.predict_segmentation(inp=frame)
			img = convert_np_to_mat(out)

			(channel_b, channel_g, channel_r) = cv2.split(img)  # ciascuna con un solo canale

			contours_b, hierarchy_b = cv2.findContours(channel_b.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			contours_g, hierarchy_g = cv2.findContours(channel_g.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			output = frame.copy()

			if len(contours_b) != 0:
				# cv2.drawContours(output, contours_b, -1, (125,125,125), 1)

				# find the biggest countour (c) by the area
				c = max(contours_b, key=cv2.contourArea)
				x, y, w, h = cv2.boundingRect(c)

				# draw the biggest contour (c) in blue
				# cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

			if len(contours_g) != 0:

				# contours_gs = sorted(contours_g, key=cv2.contourArea, reverse=True)
				# selection of the biggest area
				max_c = max(contours_g, key=cv2.contourArea)
				# draw all the contours
				# cv2.drawContours(output, contours_g, -1, (0, 255, 0), 1)
				hull = cv2.convexHull(max_c)
				area = cv2.contourArea(hull)
				# draw the convex hull contour in a blank image
				img = np.zeros([540, 980, 3], dtype=np.uint8)
				img.fill(0)
				cv2.drawContours(img, [hull], 0, (0, 0, 150), 2)

				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

				kernel_dilate = np.ones((7, 7), np.uint8)
				img = cv2.dilate(img, kernel_dilate, iterations=1)

				output = cv2.resize(frame, (640, 408), interpolation=cv2.INTER_LINEAR)
				img = cv2.resize(img, (640, 408), interpolation=cv2.INTER_LINEAR)
				frame = cv2.resize(frame, (640, 408), interpolation=cv2.INTER_LINEAR)
				cv2.imshow("im", img)

				minLineLength = 50
				maxLineGap = 0
				lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength, maxLineGap)

				if lines is not None:
					n_line = 0
					for line in lines:
						for x1, y1, x2, y2 in line:
							if n_line == 0:
								n_line += 1
								l1 = build_line([x1, y1], [x2, y2])
								cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
								x1_line1 = x1
								x2_line1 = x2
								y1_line1 = y1
								y2_line1 = y2
							if n_line > 0:
								l_tmp = build_line([x1, y1], [x2, y2])
								dist1 = abs(x2 - x2_line1)
								if dist1 > 20:
									l2 = l_tmp
									n_line += 1
									cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
									break
						if n_line == 2:
							inter_x, inter_y = intersection(l1, l2)

							if inter_y < 0:
								inter_y = 0
							if inter_y > 408:
								inter_y = 408
							if inter_x < 0:
								inter_x = 0
							if inter_x > 640:
								inter_x = 640
							print(inter_x, inter_y)
							if 200 < inter_x < 400:
								cv2.circle(output, (inter_x, inter_y), 10, (255, 0, 0), thickness=10)
							break
					# merge_fig(img_mobilenet, frame, i)

				# epsilon = 0.03 * cv2.arcLength(max_c, True)
				# approx = cv2.approxPolyDP(max_c, epsilon, True)
				# cv2.drawContours(output, [approx], 0, (0, 0, 50), 2)

				if len(max_c) > 5:
					rot_rect = cv2.fitEllipse(max_c)
					angle = rot_rect[2]
					box = cv2.boxPoints(rot_rect)
					box = np.int0(box)
					# cv2.drawContours(output, [box], 0, (0, 0, 255), 2)
					if angle > 90:
						angle = angle - 180
					print("{:.2f}".format(angle))

				cv2.imshow("output", output)


				# x1, y1, w1, h1 = cv2.boundingRect(contours_gs[0][1])
				# x2, y2, w2, h2 = cv2.boundingRect(contours_gs[0][2])
				# x3, y3, w3, h3 = cv2.boundingRect(contours_gs[0][3])
				# x4, y4, w4, h4 = cv2.boundingRect(contours_gs[0][4])

				# cv2.rectangle(output, (x3, y3), (x3 + w3, y3 + h3), (0, 155, 0), 2)
				# cv2.rectangle(output, (x4, y4), (x4 + w4, y4 + h4), (0, 255, 0), 2)
				# cv2.rectangle(output, (x2, y2), (x2 + w2, y2 + h2), (0, 105, 0), 2)
				# cv2.rectangle(output, (x1, y1), (x1 + w1, y1 + h1), (0, 55, 0), 2)

				# find the biggest countour (c) by the area
			   # c = max(contours_g, key=cv2.contourArea)
			   # x, y, w, h = cv2.boundingRect(c)
	#
			   # # draw the biggest contour (c) in green
			   # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# cv2.imshow('Original', frame)
		# cv2.imshow("CNN", img)


		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

		i += 1
	cap.release()
	cv2.destroyAllWindows()
