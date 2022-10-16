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

video_path = "Data/video.avi"

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
	object_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path="Object_Detection\\Model\\best.pt",
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
					plt.close()



			tracker_handler.update_trackers(frame)

		"""Semantic Segmentation"""
		if run_semantic:
			if frame_count % 5 == 0:
				# Segmentation runs every 5 frames

				# Convert the captured frame into RGB
				semantic_frame = Image.fromarray(rgb_frame.copy(), 'RGB')

				# Resizing into dimensions you used while training
				semantic_frame = semantic_frame.resize((480, 480))
				semantic_img_array = np.array(semantic_frame)

				# Convert to RGB
				#semantic_img_array = cv2.cvtColor(semantic_img_array, cv2.COLOR_BGR2RGB)

				# Save the input to view later
				semantic_frame_input = cv2.cvtColor(semantic_img_array.copy(), cv2.COLOR_RGB2BGR)

				# Expand dimensions to match the 4D Tensor shape.
				semantic_img_array = np.expand_dims(semantic_img_array, axis=0)
				
				# Run input image through model
				semantic_predict = semantic_segmentation_model.predict(semantic_img_array)

				# Output the mask
				semantic_output = categorical_to_mask(semantic_predict[0, :, :, :])

				# Turn into integer classes for easier use
				semantic_output = np.round(semantic_output, 3)
				semantic_output[(semantic_output == 0)] = 0
				semantic_output[(semantic_output == 0.004)] = 1
				semantic_output[(semantic_output == 0.008)] = 2
				semantic_output[(semantic_output == 0.012)] = 3

				if run_detection:
					# create binary mask to determine if theres an object there or not
					detectionMask = np.zeros((480,480))

					for index, row in dfResults.iterrows():

						# put a 1 where the object detection saw an object
						detectionMask[int(row['ymin']/2.25 - 10): int(row['ymax']/2.25 + 10), int(row['xmin']/4 - 10): int(row['xmax']/4 + 10)] = 1 
						print(int(row['xmin']/4))
						print(int(row['xmax']/4))
						print(int(row['ymin']/2.25))
						print(int(row['ymax']/2.25))

				# create semantic mask of objects, ignoring those found by the object detection
				semanticMask = np.zeros((480,480))
				semanticMask[(semantic_output == 0) | (semantic_output == 3)]  = 1
				semanticMask[(detectionMask == 1)] = 0

				# Add Semantic Overlay to image
				semanticFrameOverlay = frame.copy()
				semanticFrameOverlay = cv2.resize(semanticFrameOverlay, (480, 480))

				# apply the mask to the image, and turning any anomalies to yellow
				semanticFrameOverlay[semanticMask == 1] = [0, 255, 255]

				semanticFrameOverlayAll = semanticFrameOverlay.copy()
				semanticFrameOverlayAll[((semantic_output == 0) | (semantic_output == 3))] = [0, 255, 255]

				# Show frame
				cv2.imshow('Integration_Test_Semantic_Anomalies', semanticFrameOverlay)
				cv2.imshow('Integration_Test_Semantic', semanticFrameOverlayAll)

				if output_results:
					plt.imshow(semantic_output)
					cv2.imwrite("Results/semantic_input.jpg", semantic_frame_input)
					plt.savefig("Results/semantics_output.png")
					plt.close()

		# Draw FPS
		end = time.time()
		elapsed = end - start
		fps = frame_count / elapsed
		draw_str(frame, (100, 100), 'FPS: %.2f' % fps)

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
	run_model(output_results=False, run_detection=True, run_semantic=True)
