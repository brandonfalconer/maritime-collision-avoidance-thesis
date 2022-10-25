import itertools

import cv2
import Object_Tracking.helper_functions as hp

# Constants
from Object_Tracking.Trackers.MOSSE_filter import MOSSE

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
			for detection_bounding_box in detection_bounding_boxes:
				self.trackers.append(MOSSE(self.currentId, frame_gray, detection_bounding_box, learning_rate=self.learning_rate))
				self.currentId += 1
		else:
			trackers_matched = set()
			detections_to_remove = []
			trackers = self.trackers

			# Determine if there are multiple detections for one object
			for detection_a, detection_b in itertools.combinations(detection_bounding_boxes, 2):
				IoU = hp.intersection_over_union(detection_a, detection_b)
				if IoU > 0.5:
					# Determine if the bounding boxes are of similar width and height
					da_x1, da_y1, da_x2, da_y2 = detection_a
					da_w, da_h = da_x2 - da_x1, da_y2 - da_y1

					db_x1, db_y1, db_x2, db_y2 = detection_b
					db_w, db_h = db_x2 - db_x1, db_y2 - db_y1

					percent_diff_width = hp.get_percent_diff(da_w, db_w)
					percent_diff_height = hp.get_percent_diff(da_h, db_h)

					if percent_diff_width < .2 and percent_diff_height < .2:
						detection_bounding_boxes.remove(detection_a)
						break

			for detection_bounding_box in detection_bounding_boxes:
				for tracker in trackers:
					(x, y), (w, h) = tracker.pos, tracker.size
					tracker_bounding_box = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)

					IoU = hp.intersection_over_union(detection_bounding_box, tracker_bounding_box)

					if IoU > IOU_THRESHOLD:
						# Found a match, check if already found
						if tracker in trackers_matched:
							break

						# Update tracker position to detected position, and add to set
						if IoU < 0.5:
							tracker.update_filter(frame_gray, detection_bounding_box)
						trackers_matched.add(tracker)
						detections_to_remove.append(detection_bounding_box)
						break

			# Remove detections with matches
			for detection_to_remove in detections_to_remove:
				if detection_to_remove in detection_bounding_boxes:
					detection_bounding_boxes.remove(detection_to_remove)

			# Remove trackers with no detection match
			for tracker in self.trackers:
				if tracker not in trackers_matched:
					self.trackers.remove(tracker)

			# Create new trackers for those detections with no tracker match
			for detection_bounding_box in detection_bounding_boxes:
				self.trackers.append(MOSSE(self.currentId, frame_gray, detection_bounding_box, learning_rate=self.learning_rate))
				self.currentId += 1

			# Determine if there are multiple trackers for one object
			for tracker_a, tracker_b in itertools.combinations(self.trackers, 2):
				(x, y), (w, h) = tracker_a.pos, tracker_a.size
				tracker_a_bounding_box = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)

				(x, y), (w, h) = tracker_b.pos, tracker_b.size
				tracker_b_bounding_box = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)

				IoU = hp.intersection_over_union(tracker_a_bounding_box, tracker_b_bounding_box)

				if IoU > 0.5:
					# Determine if the bounding boxes are of similar width and height
					percent_diff_width = hp.get_percent_diff(tracker_a.size[0], tracker_b.size[0])
					percent_diff_height = hp.get_percent_diff(tracker_a.size[1], tracker_b.size[1])

					if percent_diff_width < .2 and percent_diff_height < .2:
						# So dumb right here
						if tracker_a.id > tracker_b.id:
							self.trackers.remove(tracker_a)
						else:
							self.trackers.remove(tracker_b)
						break

	def update_trackers(self, frame):
		# Get updated location of objects in subsequent frames
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		for tracker in self.trackers:
			# Determine if tracker has lost the track
			if tracker.tracking:
				tracker.untracked = 0
				tracker.update_appearance(frame_gray)
				tracker.update_motion(tracker.pos)
			else:
				# If track has been lost for too long, delete tracker
				if tracker.untracked > tracker.untracked_threshold:
					del tracker
					continue

				# Update motion model of tracker based on predicted position
				tracker.update_motion(tracker.predicted_pos)
				(x, y), (w, h) = tracker.predicted_pos, tracker.size
				(x, y) = (int(x), int(y))

				# Check area of expected motion (occlusion)
				img = cv2.getRectSubPix(frame_gray, (w, h), (x, y))
				img = tracker.preprocess(img)
				_, _, tracker.psr = tracker.correlate(img)

				# Found object from motion estimate, keep tracking with appearance
				if tracker.psr > tracker.psr_threshold:
					tracker.pos = (x, y)
					tracker.update_appearance(frame_gray)
					tracker.tracking = True

			tracker.draw_state(frame, combine_tracks=True)