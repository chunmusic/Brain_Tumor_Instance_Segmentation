import cv2
from visualize_cv2 import model, display_instances, class_names
import sys
import numpy as np

args = sys.argv
if(len(args) < 2):
	print("run command: python video_demo.py 0 or video file name")
	sys.exit(0)
name = args[1]
if(len(args[1]) == 1):
	name = int(args[1])
	
stream = cv2.VideoCapture(name)
	
while True:
	ret , frame = stream.read()
	if not ret:
		print("unable to fetch frame")
		break
	results = model.detect([frame], verbose=1)

	# Visualize results
	r = results[0]
	masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])

	cv2.imshow("masked_image",masked_image)

	# To find mask area

	# # Method#1
	# print(np.reshape(r['masks'], (-1, r['masks'].shape[-1])).astype(np.float32).sum())
	
	# # Method#2

	# for i in range(r['masks'].shape[-1]):
	# 	mask = r['masks'][:, :, i]
	# 	frame[mask] = 255
	# 	frame[~mask] = 0
	# 	unique, counts = np.unique(frame, return_counts=True)
	# 	mask_area = counts[1] / (counts[0] + counts[1]) # mask area in ratio (percent)
	# 	print(counts[1])

	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

stream.release()
cv2.destroyWindow("masked_image")