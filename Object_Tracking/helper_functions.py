"""
Test suite helper functions
"""
import sys
import cv2
import scipy.io as sio
import numpy as np


# Common metric functions
def intersection_over_union(boxA, boxB):
	"""
	Intersection over union (IoU) can be between 0 and 1, with 0 being no intersection and 1 a perfect match)
	"""
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# Area of inside intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	return interArea / float(bbox_area(boxA) + bbox_area(boxB) - interArea)


def bbox_area(box):
	return(box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def get_bbox_points(x, y, width, height):
	gt_p1 = (x, y)
	gt_p2 = (x + width, y + height)

	return gt_p1, gt_p2


def get_bbox_points_from_middle(pos, size):
	# Retrieve lower left and upper right (x,y) points with middle point and size
	p1, p2 = (int(pos[0] - 0.5 * size[0]), int(pos[1] - 0.5 * size[1])), (int(pos[0] + 0.5 * size[0]), int(pos[1] + 0.5 * size[1]))
	return p1, p2


def squared_euclidean_distance(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	x1, x2, y1, y2 = boxA[0], boxA[1], boxB[0], boxB[1]

	return (x1 - x2)**2 + (y1 - y2)**2


def get_percent_diff(a, b):
	return (abs(a - b) / ((a + b) / 2))


def get_MOTA(total_groundtruths, misses, false_positives, mismatches):
	mota = 0.0
	if total_groundtruths == 0:
		print("Warning", "No ground truth. MOTA calculation not possible")
	else:
		mota = 1.0 - float(misses + false_positives + mismatches) / float(total_groundtruths)

	return mota


def get_MOTP(total_correspondences, total_overlap):
	motp = 0.0
	if total_correspondences == 0:
		print("Warning", "No correspondences found. MOTP calculation not possible")
	else:
		motp = total_overlap / total_correspondences
	return motp


# Load GT data functions
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