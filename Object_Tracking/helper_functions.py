"""
Test suite helper functions
"""
import sys
import cv2
import scipy.io as sio
import numpy as np


def intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	return interArea / float(boxAArea + boxBArea - interArea)


def load_data_GT_Track(video_path, ground_truth_path):
	"""
	Loads data with an object tracking based ground truth

	:return: video capture, first frame, ground truth (x,y,width,height), first frame detection
	"""

	# Create a video capture object to read videos
	capture = cv2.VideoCapture(video_path)

	# Read first frame
	success, frame = capture.read()

	# Quit if unable to read the video file
	if not success:
		print('Failed to read video')
		sys.exit(1)

	gt_data = sio.loadmat(ground_truth_path)
	gt_data = gt_data['BB']
	num_objects = len(gt_data)

	first_frame_detection = []
	gt_x = []
	gt_y = []

	gt_width = []
	gt_height = []

	for x in range(num_objects):
		obj_bounding_box = gt_data[x][0][0]
		first_frame_detection.append(obj_bounding_box)

	for y in range(num_objects):
		# Append to x,y,width and height
		gt_x.append([item[0] for item in gt_data[y][0]])
		gt_y.append([item[1] for item in gt_data[y][0]])
		gt_width.append([item[2] for item in gt_data[y][0]])
		gt_height.append([item[3] for item in gt_data[y][0]])

	gt_x = np.array(gt_x).T.tolist()
	gt_y = np.array(gt_y).T.tolist()
	gt_width = np.array(gt_width).T.tolist()
	gt_height = np.array(gt_height).T.tolist()

	return capture, frame, gt_x, gt_y, gt_width, gt_height, first_frame_detection


def load_data_GT_TrackAnalysis(video_path, ground_truth_path):
	"""
	Loads data with an object detection based ground truth (updated less frequently than tracking GT)

	:return: video capture, first frame, ground truth (x,y,width,height), first frame detection
	"""

	# Create a video capture object to read videos
	capture = cv2.VideoCapture(video_path)

	# Read first frame
	success, frame = capture.read()

	# Quit if unable to read the video file
	if not success:
		print('Failed to read video')
		sys.exit(1)

	gt_data = sio.loadmat(ground_truth_path)
	gt_data = gt_data['TrackAnalysis'][0][0]

	gt_width = gt_data[2]
	gt_height = gt_data[3]

	gt_x = gt_data[0] - (gt_width / 2)
	gt_y = gt_data[1] - (gt_height / 2)

	first_frame_detection = []
	for x in range(len(gt_data[0][0])):
		obj_bounding_box = [gt_x[0][x], gt_y[0][x], gt_width[0][x], gt_height[0][x]]
		first_frame_detection.append(obj_bounding_box)

	return capture, frame, gt_x, gt_y, gt_width, gt_height, first_frame_detection