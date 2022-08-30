import cv2  # openCV 4.5.1
import numpy as np
from numpy import prod
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
from skimage.io import imread
from skimage.transform import resize
from PIL import Image, ImageFont, ImageDraw  # add caption by using custom font

from collections import deque

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # Set the GPU 2 to use

NMS_THRESHOLD=0.3
MIN_CONFIDENCE=0.2


def pedestrian_detection(image, model, layer_name, personidz=0):
	(H, W) = image.shape[:2]
	results = []


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > MIN_CONFIDENCE:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			res = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(res)
	# return the list of results
	return results


# bi

IMG_SIZE = 224
# preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
base = tf.keras.applications.MobileNetV3Small(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
                                              weights='imagenet')
# base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet', include_top=False)
base_model = tf.keras.Sequential([
    base,
    tf.keras.layers.Flatten()
])

base_model.trainable = False
# base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(input_shape = (IMG_SIZE, IMG_SIZE, 3), weights='imagenet', include_top=False)
# base_model=keras.applications.mobilenet.MobileNet(input_shape=(160, 160, 3), include_top=False, weights='imagenet', classes=2)


model = keras.models.load_model('../model/new_model.h5')
model.trainable = False

input_path = 'input_어린이.mp4'
output_path = 'output_어린이.mp4'

# detect
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

cv2_model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_name = cv2_model.getLayerNames()
layer_name = [layer_name[i - 1] for i in cv2_model.getUnconnectedOutLayers()]

vid = cv2.VideoCapture(input_path)
fps = vid.get(cv2.CAP_PROP_FPS)  # recognize frames per secone(fps) of input_path video file.
print(f'fps : {fps}')  # print fps.
fps = 30
writer = None
(W, H) = (None, None)
i = 0  # number of seconds in video = The number of times that how many operated while loop .
Q = deque(maxlen=128)

video_frm_ar = np.zeros((1, int(fps), IMG_SIZE, IMG_SIZE), dtype=np.float64)  # frames
frame_counter = 0  # frame number in 1 second. 1~30
frame_list = []
preds = None
maxprob = None

# . While loop : Until the end of input video, it read frame, extract features, predict violence True or False.
# ----- Reshape & Save frame img as (30, 160, 160, 3) Numpy array  -----
grabbed, frm = vid.read()

if W is None or H is None:  # W: width, H: height of frame img
    (H, W) = frm.shape[:2]
frame = cv2.resize(frm, (IMG_SIZE, IMG_SIZE))
pre_frm = frame.copy()
# pre detection mask
results = pedestrian_detection(pre_frm, cv2_model, layer_name, personidz=LABELS.index("person"))
mask = np.zeros(frame.shape[:2], np.uint8)
for res in results:
    polygon = np.array([[res[1][0] - 5, res[1][1] - 5], [res[1][2] + 5, res[1][1] - 5], [res[1][2] + 5, res[1][3] + 5],
                        [res[1][0] - 5, res[1][3] + 5]])
    cv2.fillPoly(mask, [polygon], 1)
pre_mask = mask.astype(bool)

while True:
    frame_counter += 1
    grabbed, frm = vid.read()  # read each frame img. grabbed=True, frm=frm img. ex: (240, 320, 3)

    if not grabbed:
        print('There is no frame. Streaming ends.')
        break

    if W is None or H is None:  # W: width, H: height of frame img
        (H, W) = frm.shape[:2]

    output = frm.copy()  # It is necessary for streaming captioned output video, and to save that.

    frame = cv2.resize(frm, (IMG_SIZE, IMG_SIZE))  # > Resize frame img array to (160, 160, 3)

    # detection mask

    results = pedestrian_detection(frame, cv2_model, layer_name, personidz=LABELS.index("person"))
    mask = np.zeros(frame.shape[:2], np.uint8)
    for res in results:
        polygon = np.array(
            [[res[1][0] - 10, res[1][1] - 10], [res[1][2] + 10, res[1][1] - 10], [res[1][2] + 10, res[1][3] + 10],
             [res[1][0] - 10, res[1][3] + 10]])
        cv2.fillPoly(mask, [polygon], 1)
    cur_mask = mask.astype(bool)

    # optical flow
    hsv = np.zeros_like(pre_frm)
    hsv[..., 1] = 255
    hsv = np.array(hsv, dtype=np.float32)
    pre_frm = np.array(pre_frm, dtype=np.float32)
    cur_frm = np.array(frame, dtype=np.float32)
    pre_frm = cv2.cvtColor(pre_frm, cv2.COLOR_BGR2GRAY)
    cur_frm = cv2.cvtColor(cur_frm, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(pre_frm, cur_frm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # mask
    rgb = rgb * (cur_mask[:, :, np.newaxis] + pre_mask[:, :, np.newaxis])

    frame_list.append(rgb)  # Append each frame img Numpy array : element is (160, 160, 3) Numpy array.
    pre_frm = frame.copy()
    pre_mask = cur_mask.copy()

    if frame_counter >= fps:  # fps=30 et al
        # . ----- we'll predict violence True or False every 30 frame -----
        # . ----- Insert (1, 30, 160, 160, 3) Numpy array to LSTM model ---
        # . ----- We'll renew predict result caption on output video every 1 second. -----
        # 30-element-appended list -> Transform to Numpy array -> Predict -> Initialize list (repeat)
        frame_ar = np.array(frame_list, dtype=np.float16)  # > (30, 160, 160, 3)
        frame_list = []  # Initialize frame list when frame_counter is same or exceed 30, after transforming to Numpy array.

        if (np.max(frame_ar) > 1):
            frame_ar = frame_ar / 255.0  # Scaling RGB value in Numpy array

        pred_imgarr = base_model.predict(
            frame_ar)  # > Extract features from each frame img by using MobileNet. (30, 5, 5, 1024)
        pred_imgarr_dim = pred_imgarr.reshape(1, pred_imgarr.shape[0], 7 * 7 * 576)  # > (1, 30, 25600)

        preds = model.predict(pred_imgarr_dim)  # > (True, 0.99) : (Violence True or False, Probability of Violence)
        print(f'preds:{preds}')
        #         Q.append(preds)

        #         # Predict Result : Average of Violence probability in last 5 second
        #         if i < 5:
        #             results = np.array(Q)[:i].mean(axis=0)
        #         else:
        #             results = np.array(Q)[(i - 5):i].mean(axis=0)

        #         print(f'Results = {results}')  # > ex : (0.6, 0.650)
        #         maxprob=np.max(results) #> Select Maximum Probability

        maxprob = np.max(preds)  # > Select Maximum Probability
        print(f'Maximum Probability : {maxprob}')
        print('')

        #         rest=1-maxprob # Probability of Non-Violence
        #         diff=maxprob-rest # Difference between Probability of Violence and Non-Violence's
        th = 0.65

        #         if diff>0.60:
        #             th=diff # ?? What is supporting basis?

        frame_counter = 0  # > Initialize frame_counter to 0
        i += 1  # > 1 second elapsed

        # When frame_counter>=30, Initialize frame_counter to 0, and repeat above while loop.

    # ----- Setting caption option of output video -----
    # Renewed caption is added every 30 frames(if fps=30, it means 1 second.)
    font_size = 160
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    if preds is not None and maxprob is not None:
        if (preds[0][0]) < th:  # > if violence probability < th, Violence=False (Normal, Green Caption)
            text1_1 = 'Normal'
            text1_2 = '{:.2f}%'.format(preds[0][1] * 100)
            img_pil = Image.fromarray(output)
            draw = ImageDraw.Draw(img_pil)
            draw.text((int(0.025 * W), int(0.025 * H)), text1_1, font=font, fill=(0, 255, 0, 0))
            draw.text((int(0.025 * W), int(0.105 * H)), text1_2, font=font, fill=(0, 255, 0, 0))
            output = np.array(img_pil)

        else:  # > if violence probability > th, Violence=True (Violence Alert!, Red Caption)
            text2_1 = 'Abuse Detected'
            text2_2 = '{:.2f}%'.format(maxprob * 100)
            img_pil = Image.fromarray(output)
            draw = ImageDraw.Draw(img_pil)
            draw.text((int(0.025 * W), int(0.025 * H)), text2_1, font=font, fill=(0, 0, 255, 0))
            draw.text((int(0.025 * W), int(0.105 * H)), text2_2, font=font, fill=(0, 0, 255, 0))
            output = np.array(img_pil)

    # Save captioned video file by using 'writer'
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)

    # cv2.imshow('This is output', output)  # View output in new Window.
    writer.write(output)  # Save output in output_path

    # key = cv2.waitKey(round(1000 / 30))  # time gap of frame and next frame
    # if key == 27:  # If you press ESC key, While loop will be breaked and output file will be saved.
    #    print('ESC is pressed. Video recording ends.')
    #    break

print('Video recording ends. Release Memory.')  # Output file will be saved.
writer.release()
vid.release()
cv2.destroyAllWindows()