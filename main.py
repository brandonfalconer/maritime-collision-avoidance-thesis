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

from Object_Tracking.multiple_object_tracker import MOT

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


def run_model(output_results=False, run_detection=True, run_semantic=True):
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
		rgb_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

		""" Object Detection and Object Tracking"""
		if run_detection:
			detection_bounding_boxes = []

			# Initialise tracker handler
			if frame_count == 0:
				detection_results = object_detection_model(rgb_frame.copy())
				dfResults = detection_results.pandas().xyxy[0]

				for index, row in dfResults.iterrows():
					detection_bounding_boxes.append(
						(int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])))

				tracker_handler = MOT(frame, detection_bounding_boxes, 0.125)

			if frame_count % 20 == 0:
				# Detection runs every 5 frames
				detection_results = object_detection_model(rgb_frame.copy())
				dfResults = detection_results.pandas().xyxy[0]

				for index, row in dfResults.iterrows():
					detection_bounding_boxes.append(
						(int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])))

				tracker_handler.update_tracker_list(detection_bounding_boxes)

			# Draw detections as blue bb
			for index, row in dfResults.iterrows():
				cv2.rectangle(frame, pt1=(int(row['xmin']), int(row['ymin'])), pt2=(int(row['xmax']), int(row['ymax'])), color=(255, 0, 0), thickness=2)

				if output_results:
					fig, ax = plt.subplots(figsize=(16, 12))
					ax.imshow(detection_results.render()[0])
					plt.savefig("Results/detection_output.png")



			tracker_handler.update_trackers(frame)

		"""Semantic Segmentation"""
		if run_semantic:
			if frame_count % 5 == 0:
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

				if output_results:
					plt.imshow(output)
					semantic_frame.save("Results/semantic_input.jpg")
					plt.savefig("Results/semantics_output.png")

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
	# Set output_results to true, to save files of detection/segmentation output
	run_model(output_results=False, run_detection=True, run_semantic=False)
