#!/usr/bin/env python
import numpy as np
import cv2

"""
Object tracking filter utilising the MOSSE filter and the Kalman filter.

Minimum Output Some of Squared Error (MOSSE) Tracker created by David Bolme, et al. 
Paper source: https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf
Explanation of Paper: https://mountainscholar.org/bitstream/handle/10217/173486/Sidhu_colostate_0053N_13486.pdf

Based off the OpenCV implementation: https://docs.opencv.org/3.4/d0/d02/classcv_1_1TrackerMOSSE.html TrackerMOSSE()

Edited by Brandon Falconer 27/9/2022
"""

# Constants
# epsilon, the smallest possible positive floating-point number (to avoid division by 0)
EPS = 1e-5


def rnd_warp(a):
	"""
	# Use affine transformations of a single template to create more training samples.

	Affine transformations such as scale and rotation can be applied to the template obtained from the first frame of a video.
	The affine transformations are used to initialize the filter.
	Due to slight variations of an object in consecutive frames, small affine transformations are applied.

	:param a: tracking image
	:return: random affine transformation
	"""

	h, w = a.shape[:2]
	T = np.zeros((2, 3))
	coef = 0.2
	ang = (np.random.rand() - 0.5) * coef
	c, s = np.cos(ang), np.sin(ang)
	T[:2, :2] = [[c, -s], [s, c]]
	T[:2, :2] += (np.random.rand(2, 2) - 0.5) * coef
	c = (w / 2, h / 2)
	T[:, 2] = c - np.dot(T[:2, :2], c)
	return cv2.warpAffine(a, T, (w, h), borderMode=cv2.BORDER_REFLECT)


def divSpec(A, B):
	# Divide two complex multidimensional arrays

	# Multidimensional array slicing
	Ar, Ai = A[..., 0], A[..., 1]
	Br, Bi = B[..., 0], B[..., 1]
	C = (Ar + 1j * Ai) / (Br + 1j * Bi)
	# Stack arrays in sequence depth wise (along third axis)
	C = np.dstack([np.real(C), np.imag(C)]).copy()
	return C


def draw_str(dst, target, s):
	x, y = target
	cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


class Tracker:
	def __init__(self, id, frame, rect, psr_threshold=8, learning_rate=0.125, sigma=2):
		"""
		Initialise the tracker
		with a specified rectangle over the object in which to track

		self.G is the synthetic target image in the fourier domain
		self.H is the complex conjugate of the filter in the frequency domain
		self.win is the preprocessed template in the Fourier Domain

		:param frame: greyscale frame of video to initialise the tracker
		:param rect: selected tracking rect containing the object to be tracked
		:param psr_threshold: peak-to-sidelobe ratio threshold to decide whether to continue tracking
		:param learning_rate: interp factor
		:param sigma: gaussian blur border type (pixel extrapolation method)
		"""

		self.id = id
		self.psr_threshold = psr_threshold
		self.learning_rate = learning_rate

		self.predicted_motion = []
		self.predicted_pos = (0, 0)

		# Retrieve the position and size of the initial tracking template
		x1, y1, x2, y2 = rect

		# Returns the minimum number N that is greater than or equal to vecsize so that the DFT of a vector of size N
		# can be processed efficiently
		w, h = map(cv2.getOptimalDFTSize, [x2 - x1, y2 - y1])
		x1, y1 = (x1 + x2 - w) // 2, (y1 + y2 - h) // 2
		self.pos = x, y = x1 + 0.5 * (w - 1), y1 + 0.5 * (h - 1)
		self.size = w, h
		img = cv2.getRectSubPix(frame, (w, h), (x, y))  # Retrieves a pixel rectangle from an image with sub-pixel accuracy

		# Create a synthetic target generated with the Gaussian peak centered on the centre of the object
		self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)  # Computes a Hanning window coefficients in two dimensions
		g = np.zeros((h, w), np.float32)
		g[h // 2, w // 2] = 1
		g = cv2.GaussianBlur(g, (-1, -1), sigma)  # Blurs an image using a Gaussian filter
		g /= g.max()

		# Convert the synthetic target image to the Fourier Domain
		self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
		# Return an array of zeros with the same shape and type as a given array
		self.H1 = np.zeros_like(self.G)
		self.H2 = np.zeros_like(self.G)
		for _i in range(128):
			# Perform random affine transformations for filter initialization
			a = self.preprocess(rnd_warp(img))
			A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
			# Performs the per-element multiplication of two Fourier spectrums
			self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
			self.H2 += cv2.mulSpectrums(A, A, 0, conjB=True)

		# Create kalman filter with state dimension (x, y, width, height) and measurement dimension (x, y)
		self.kf = cv2.KalmanFilter(4, 2)
		self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
		self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
		# Covariance Matrix (Symmetric)
		self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.01

		# Kalman filter does not have initial state setting, use initial state to correct/measure in update step
		self.initial_state = np.asarray(self.pos, dtype=np.float32)

		self._update_kernel()
		self.update_appearance(frame)
		self.update_motion(self.pos)

	def update_appearance(self, frame):
		"""
		Pre-process and update the input frame performed before initialisation and tracking

		:param frame: tracking input frame
		:return: the updated position of the tracking window
		"""

		# Crop a template of n by n pixels
		(x, y), (w, h) = self.pos, self.size
		self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
		img = self.preprocess(img)

		self.last_resp, (dx, dy), self.psr = self.correlate(img)
		self.tracking = self.psr > self.psr_threshold
		if not self.tracking:
			return

		self.pos = x + dx, y + dy
		self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
		img = self.preprocess(img)

		# Compute the discrete fourier transform (dft) of the image
		A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

		# Perform per-element multiplication of two Fourier spectrum's
		H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
		H2 = cv2.mulSpectrums(A, A, 0, conjB=True)

		self.H1 = self.H1 * (1.0 - self.learning_rate) + H1 * self.learning_rate
		self.H2 = self.H2 * (1.0 - self.learning_rate) + H2 * self.learning_rate
		self._update_kernel()

	def update_motion(self, pos):
		# Update the kf motion object_detection_model
		correct_pos = np.array(
			[[np.float32(pos[0]) - self.initial_state[0]], [np.float32(pos[1]) - self.initial_state[1]]])
		self.kf.correct(correct_pos)
		predict_pos = self.kf.predict()
		predict_pos[0] = predict_pos[0] + self.initial_state[0]
		predict_pos[1] = predict_pos[1] + self.initial_state[1]

		self.predicted_motion.append((int(predict_pos[0]), int(predict_pos[1])))
		self.predicted_pos = (int(predict_pos[0]), int(predict_pos[1]))

	def preprocess(self, img):
		# Apply log transformation to reduce lighting effects and enhance contract, making high contrast features
		# available for the filter to initialise on
		img = np.log(np.float32(img) + 1.0)

		# Normalise to get a mean of zero and a normal of one (reducing the effect of change in illumination)
		img = (img - img.mean()) / (img.std() + EPS)
		return img * self.win

	def correlate(self, img):
		"""
		Images are projected in a correlation space of size N dimension (width * height)

		Image is unrolled to one dimensional vector, and mean is computed and subtracted from the vector. The vector
		is then normalised. Correlation space is a unit length sphere, where taking the dot product of two vectors in
		correlation space is taking the cosine of the angles between the vectors.

		:param img: image to be correlated
		:return: peak-to-sidelobe ratio
		"""

		# Performs the per-element multiplication of two Fourier spectrum's
		C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
		resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
		h, w = resp.shape
		# Find the maximum and minimum elements and their positions
		_, min_val, _, (min_x, min_y) = cv2.minMaxLoc(resp)
		side_resp = resp.copy()
		cv2.rectangle(side_resp, (min_x - 5, min_y - 5), (min_x + 5, min_y + 5), 0, -1)
		smean, sstd = side_resp.mean(), side_resp.std()
		psr = (min_val - smean) / (sstd + EPS)
		return resp, (min_x - w // 2, min_y - h // 2), psr

	def _update_kernel(self):
		self.H = divSpec(self.H1, self.H2)
		self.H[..., 1] *= -1

	def __eq__(self, other):
		return isinstance(other, Tracker) and self.id == other.id

	def __hash__(self):
		return hash(self.id)

	def draw_state(self, vis):
		"""
		Draw the bounding box around the current target location and the current psr

		:param vis: current image to draw on
		:return: None
		"""
		(x, y), (w, h) = self.pos, self.size
		x1, y1, x2, y2 = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)

		# Draw string self ID
		draw_str(vis, (x1, y2 + 16), 'ID: %d' % self.id)

		# Draw tracking rectangle
		cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))

		# Graphically show if track is accurate
		if self.tracking:
			cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
		else:
			# Draw a cross through the bounding box to represent a bad track
			cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
			cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))

		# Draw string peak-to-sidelobe ratio
		draw_str(vis, (int(x1 + 64), y2 + 16), 'PSR: %.2f' % self.psr)

		# Draw predicted motion object_detection_model
		'''
		cv2.rectangle(vis, pt1=(self.predicted_pos[0] - self.size[0] // 2, self.predicted_pos[1] - self.size[1] // 2),
					  pt2=(self.predicted_pos[0] + self.size[0] // 2, self.predicted_pos[1] + self.size[1] // 2),
					  color=(0, 255, 0), thickness = 2)'''

		#print("Predicted x diff: " + str(self.pos[0] - self.predicted_pos[0]))
		#print("Predicted y diff: " + str(self.pos[1] - self.predicted_pos[1]))