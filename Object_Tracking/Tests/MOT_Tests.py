import sys
import time
import cv2
import random
import helper_functions as hp

# Constants
from Trackers.MOSSE_Kalman_MOT import MOSSE

current_video_path = "C:\\Users\\Johnson\\OneDrive - Queensland University of Technology\\Thesis\\pythonModel\\" \
					 "Data\\Singapore\\VIS_Onshore\\Videos\\MVI_1640_VIS.avi"
current_ground_truth_path = "C:\\Users\\Johnson\\OneDrive - Queensland University of Technology\\Thesis\\pythonModel\\" \
							"Data\\Singapore\\VIS_Onshore\\TrackGT\\MVI_1640_VIS_TrackGT_TrackingGT.mat"


def test_MOSSE_tracker(video_path, ground_truth_path, learning_rate=0.125):
	# Retrieve testing data
	capture, frame, gt_x, gt_y, gt_width, gt_height, first_frame_detection = \
		hp.load_data_GT_TrackAnalysis(video_path, ground_truth_path)

	# Select boxes
	gt_bounding_boxes, gt_colors, trackers, tracker_boxes, tracker_colors = [], [], [], [], []

	num_objects = len(first_frame_detection)
	trackers_not_initialised = []

	for i in range(num_objects):
		if sum(first_frame_detection[i]) > 0:
			gt_bounding_boxes.append(tuple(first_frame_detection[i]))
		else:
			trackers_not_initialised.append(i)

	print("Selected multiple objects with initial bounding boxes: {}".format(gt_bounding_boxes))

	# Initialize MOSSE tracker
	curr_id = 1
	for bbox in gt_bounding_boxes:
		# Append the tracker
		rect = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		trackers.append(MOSSE(curr_id, frame_gray, rect, learning_rate=learning_rate))
		curr_id += 1

		gt_colors.append((random.randint(100, 255), 0, 0))
		tracker_colors.append((0, 0, (random.randint(100, 255))))

	frame_count, IoUCount, totalIoU = 0, 0, 0

	start_time = time.time()

	# Process video and track objects
	while capture.isOpened():
		time.sleep(0.1)
		success, frame = capture.read()
		if not success:
			break

		# Check if a new object has entered the scene
		for i in trackers_not_initialised:
			if gt_x[frame_count][i] > 0:
				# Append the tracker
				rect = int(gt_x[frame_count][i]), int(gt_y[frame_count][i]), \
					   int(gt_x[frame_count][i] + gt_width[frame_count][i]), int(gt_y[frame_count][i] + gt_height[frame_count][i])

				print("Selected new object with bounding box: {}".format(rect))
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				trackers.append(MOSSE(curr_id, frame_gray, rect, learning_rate=learning_rate))
				curr_id += 1

				gt_colors.append((random.randint(100, 255), 0, 0))
				tracker_colors.append((0, 0, (random.randint(100, 255))))

				trackers_not_initialised.remove(i)

		# Get updated location of objects in subsequent frames
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		tracker_boxes = []
		for tracker in trackers:
			# Determine if tracker has lost the track
			if tracker.tracking:
				tracker.update_appearance(frame_gray)
				tracker.update_motion(tracker.pos)
				(x, y), (w, h) = tracker.pos, tracker.size
				tracker_boxes.append((x, y, w, h))
				tracker.draw_state(frame)
			else:
				tracker.update_motion(tracker.predicted_pos)
				(x, y), (w, h) = tracker.predicted_pos, tracker.size
				tracker_boxes.append((x, y, w, h))
				tracker.draw_state(frame)

				(x, y), (w, h) = tracker.pos, tracker.size
				tracker.last_img = img = cv2.getRectSubPix(frame_gray, (w, h), (x, y))
				img = tracker._preprocess(img)
				_, _, psr = tracker.correlate(img)
				if psr > 8:
					tracker.tracking = True

		# If ground truth data does not exist beyond the current frame_count frame, break
		if frame_count >= len(gt_x):
			break

		# Draw the bounding boxes of the tracked objects
		# GT are shades of blue, tracker bbox are shades of red
		for j, box in enumerate(tracker_boxes):
			# If the ground truth object does not exist in the current frame, continue
			if gt_x[frame_count][j] <= 0:
				#print(gt_x[frame_count][j])
				continue

			# Create tracker bounding box
			tracker_pos = trackers[j].pos
			tracker_size = trackers[j].size
			track_p1, track_p2 = (int(tracker_pos[0] - 0.5 * tracker_size[0]), int(tracker_pos[1] - 0.5 * tracker_size[1])), \
								 (int(tracker_pos[0] + 0.5 * tracker_size[0]), int(tracker_pos[1] + 0.5 * tracker_size[1]))

			# Create GT bounding box
			gt_p1 = (int(gt_x[frame_count][j]), int(gt_y[frame_count][j]))
			gt_p2 = (int(gt_x[frame_count][j] + gt_width[frame_count][j]),
					 int(gt_y[frame_count][j] + (gt_height[frame_count][j])))
			cv2.rectangle(frame, gt_p1, gt_p2, gt_colors[j], thickness=1, lineType=1)

			totalIoU += hp.intersection_over_union((track_p1 + track_p2), (gt_p1 + gt_p2))
			IoUCount += 1

		frame_count += 1

		# Show frame
		cv2.imshow('MultiTracker', frame)

		# Quit on ESC button
		if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
			sys.exit(1)

	end = time.time()

	# Time elapsed
	elapsed = end - start_time
	fps = frame_count / elapsed

	averageIoU = totalIoU / IoUCount
	print("Tracker Type: MOSSE \nAverage IoU: " + str(averageIoU))
	print("Elapsed Time: " + str(elapsed))
	print("FPS: " + str(fps) + "\n")

	return averageIoU, fps


def test_learning_rate():
	# Tests using internal MOSSE tracker
	lr_to_test = [0.01, 0.075, 0.125, 0.175, 0.3]

	for i in range(len(lr_to_test)):
		print("Testing with lr of: " + str(lr_to_test[i]))
		IoU, _ = test_MOSSE_tracker(current_video_path, current_ground_truth_path, learning_rate=lr_to_test[i])

	print("---Summary---")


if __name__ == '__main__':
	#test_learning_rate()

	IoU, fps = test_MOSSE_tracker(current_video_path, current_ground_truth_path, learning_rate=0.00000000000001)