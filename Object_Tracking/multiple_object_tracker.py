import sys
import time
import cv2
import random
import Object_Tracking.helper_functions as hp

# Constants
from Object_Tracking.Model.MOSSE_Kalman_filter import Tracker

IOU_THRESHOLD = 0.5

class MOT:
	def __init__(self, frame, object_detections, learning_rate):
		self.learning_rate = learning_rate
		self.frame = frame
		self.trackers = []
		self.currentId = 0

		self.update_tracker_list(object_detections)

	def update_tracker_list(self, detection_bounding_boxes):
		frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

		if len(self.trackers) == 0:
			print("Original detections")
			print(detection_bounding_boxes)
			for detection_bounding_box in detection_bounding_boxes:
				self.trackers.append(Tracker(self.currentId, frame_gray, detection_bounding_box, learning_rate=self.learning_rate))
				self.currentId += 1
		else:
			print("New detections")
			print(detection_bounding_boxes)
			trackers_matched = set()
			detections_to_remove = []
			trackers = self.trackers.copy()

			print(len(detection_bounding_boxes))
			index = 0
			for detection_bounding_box in detection_bounding_boxes:
				print("INDEX: " + str(index))
				for tracker in trackers:

					(x, y), (w, h) = tracker.pos, tracker.size
					tracker_bounding_box = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)

					IoU = hp.intersection_over_union(detection_bounding_box, tracker_bounding_box)
					print("Tracker ID: " + str(tracker.id) + " Detect BB: " + str(detection_bounding_box) + " Track BB: " + str(tracker_bounding_box) + ", IoU: " + str(IoU))

					if IoU > IOU_THRESHOLD:
						print("Tracker match with ID: " + str(tracker.id))
						# Found a match, check if already found
						if tracker in trackers_matched:
							break

						# Update tracker position to detected position, and add to set
						d_x1, d_y1, d_x2, d_y2 = detection_bounding_box
						d_w, d_h = d_x2 - d_x1, d_y2 - d_y1

						print(detection_bounding_box)

						"""
						tracker.pos = d_x1 + 0.5 * (d_w - 1), d_y1 + 0.5 * (d_h - 1)
						tracker.size = d_w, d_h
						tracker.win = cv2.createHanningWindow((d_w, d_h), cv2.CV_32F)
						tracker.last_img = cv2.getRectSubPix(self.frame, (w, h), tracker.pos)
						"""

						trackers_matched.add(tracker)
						detections_to_remove.append(detection_bounding_box)
						#detection_bounding_boxes.pop(index)
						#trackers.remove(tracker)
						break
				index += 1

			# Remove detections with matches
			for detection_to_remove in detections_to_remove:
				if detection_to_remove in detection_bounding_boxes:
					detection_bounding_boxes.remove(detection_to_remove)

			# Remove trackers with no detection match
			for tracker in self.trackers:
				if tracker not in trackers_matched:
					self.trackers.remove(tracker)

			# Create new trackers for those detections with no tracker match
			print("UPDATING")
			print(detection_bounding_boxes)
			for detection_bounding_box in detection_bounding_boxes:
				self.trackers.append(Tracker(self.currentId, frame_gray, detection_bounding_box, learning_rate=self.learning_rate))
				self.currentId += 1


	def update_trackers(self, frame):
		# Get updated location of objects in subsequent frames
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		for tracker in self.trackers:
			# Determine if tracker has lost the track
			if tracker.tracking:
				# Update tracker appearance filter and motion model
				tracker.update_appearance(frame_gray)
				tracker.update_motion(tracker.pos)

				# Draw tracker to frame
				tracker.draw_state(frame)
			else:

				# Appearance filter has failed, continue to check the region for correlations in-case of occlusion
				print("Stopped tracking: " + str(tracker.id))

				tracker.update_motion(tracker.predicted_pos)

				(x, y), (w, h) = tracker.pos, tracker.size
				tracker.last_img = img = cv2.getRectSubPix(frame_gray, (w, h), (x, y))
				img = tracker.preprocess(img)
				_, _, psr = tracker.correlate(img)
				if psr > 8:
					tracker.tracking = True