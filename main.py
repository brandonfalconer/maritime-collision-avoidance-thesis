import sys
import time

import cv2
import torch

import numpy as np
import pandas as pd

import tensorflow as tf
from PIL import Image
import glob
import matplotlib.pyplot as plt

video_path = "Data/MVI_1469_VIS.avi"

# Classes
names = {
	0: "none",
	1: "ferry",
	2: "buoy",
	3: "vessel/ship",
	4: "speed_boat",
	5: "boat",
	6: "kayak",
	7: "sail_boat",
	8: "swimming_person",
	9: "flying_bird/plane",
	10: "other"
}


def draw_str(dst, target, s):
	x, y = target
	cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


def categorical_to_mask(im):
	mask = tf.dtypes.cast(tf.argmax(im, axis=2), 'float32') / 255.0
	return mask


def load_files(path, target_size, scale_factor):
	image_list = []
	filenames = glob.glob(path)
	filenames.sort()
	for filename in filenames:
		im = Image.open(filename)
		w, h = im.size
		im = im.resize((target_size, target_size))
		im = np.asarray(im) / scale_factor
		image_list.append(im)
	return np.asarray(image_list)


def run_model(results=False):
	# Create a video capture object to read videos
	capture = cv2.VideoCapture(video_path)

	# Read first frame
	success, frame = capture.read()

	# Quit if unable to read the video file
	if not success:
		print('Failed to read video')
		sys.exit(1)

	frame_count = 0

	# Configs
	torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
	tf.get_logger().setLevel('INFO')

	# Load models
	object_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path="Object_Detection\Model\smd1_best.pt",
											force_reload=True)
	semantic_segmentation_model = tf.keras.models.load_model('Semantic_Segmentation/Model', compile=False)

	# Initialise starting time for average FPS
	start = time.time()

	# Process video
	while capture.isOpened():
		success, frame = capture.read()
		if not success:
			break

		# Convert the captured frame into RGB
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Object Detection
		if frame_count % 5 == 0 or frame_count == 0:
			# Detection runs every 5 frames
			results = object_detection_model(rgb_frame.copy())
			print("Frame:" + str(frame_count))
			print("Number of objects: " + str(len(results.pandas().xyxy[0])))

			if results:
				fig, ax = plt.subplots(figsize=(16, 12))
				ax.imshow(results.render()[0])
				plt.savefig("Results/detection_output.png")

		# Semantic Segmentation
		if frame_count % 5 == 0 or frame_count == 0:
			# Segmentation runs every 5 frames

			# Convert the captured frame into RGB
			semantic_frame = Image.fromarray(rgb_frame.copy(), 'RGB')

			# Resizing into dimensions you used while training
			semantic_frame = semantic_frame.resize((128, 128))
			semantic_img_array = np.array(semantic_frame)

			# Expand dimensions to match the 4D Tensor shape.
			semantic_img_array = np.expand_dims(semantic_img_array, axis=0)

			predict = semantic_segmentation_model.predict(semantic_img_array)
			output = categorical_to_mask(predict[0, :, :, :])

			if results:
				semantic_plot = plt.imshow(output)
				semantic_frame.save("Results/semantic_input.jpg")
				plt.savefig("Results/semantics_output.png")

		# Object Tracking TODO

		# Show results
		dfResults = results.pandas().xyxy[0]
		for index, row in dfResults.iterrows():
			cv2.rectangle(frame, pt1=(int(row['xmin']), int(row['ymin'])), pt2=(int(row['xmax']), int(row['ymax'])), color=(255, 0, 0), thickness=2)

		# Draw FPS
		end = time.time()
		elapsed = end - start
		fps = frame_count / elapsed
		draw_str(frame, (100, 100), 'FPS: %.2f' % fps)

		# Show frame
		cv2.imshow('Integration_Test', frame)

		frame_count += 1

		# Quit on ESC button
		key = cv2.waitKey(1)
		if key & 0xFF == 27:  # Esc pressed
			sys.exit(1)
		if key == ord('p'):
			cv2.waitKey(-1)  # wait until any key is pressed

	capture.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	# Set results to true, to save files of detection/segmentation output
	run_model(False)
