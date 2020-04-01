import os
import numpy as np

def reduce_by_overlap(detections, indices, overlap_thresh):
	discard = []
	for i in range(len(indices) - 1):
		for j in range(i+1, len(indices)):
			index1 = indices[i]
			index2 =  indices[j]
			_, _, conf1, x1min, y1min, x1max, y1max = detections[index1]
			_, _, conf2, x2min, y2min, x2max, y2max = detections[index2]
			xmin = max(x1min, x2min)
			ymin = max(y1min, y2min)
			xmax = min(x1max, x2max) 
			ymax = min(y1max, y2max)
			iw = max(xmax - xmin, 0)
			ih = max(ymax - ymin, 0)
			inters = iw * ih
			uni = (x1max - x1min) * (y1max - y1min) + (x2max - x2min) * (y2max - y2min) - inters
			overlap = inters * 1.0 / uni
			if(overlap > overlap_thresh):
				if(conf1 > conf2):
					discard.append(index2)
				else:
					discard.append(index1)
	return [index for index in indices if index not in discard]

#CHANGE TO YOUR OWN PATH
detection_results_root = '/detection_results' #where the .npy object detection files are saved
top_detections_root = '/top_detections'

detections = os.listdir(detection_results_root)
total_cls = 15
frame_per_video = 80
confidence_thresh = 0.9
overlap_thresh = 0.7
max_number_of_cls = 2
single_detection_each_class = False
class_count_dic = {}

for detection in detections:
	print detection
	detection_file = os.path.join(detection_results_root, detection)
	detection_npy = np.load(detection_file)

	#filter by confidence
	confidence_threshed = [[] for _ in range(frame_per_video + 1)]
	for index, single_detection in enumerate(detection_npy):
		(frame, _, confidence, _, _, _, _) = single_detection
		frame = int(frame)
		if(confidence > confidence_thresh):
			confidence_threshed[frame].append(index)

	#filter by overlap
	overlap_threshed = [[] for _ in range(total_cls + 1)]
	for frame in range(1, frame_per_video + 1):
		if(len(confidence_threshed[frame]) == 0):
			continue
		if(len(confidence_threshed[frame]) > 1):
			confidence_threshed[frame] = reduce_by_overlap(detection_npy, confidence_threshed[frame], overlap_thresh)
		for index in confidence_threshed[frame]:
			(_, cls, _, _, _, _, _) = detection_npy[index]
			cls = int(cls)
			overlap_threshed[cls].append(index)
	
	#whether only keep the top detection for each class
	single_detection_threshed = [[] for _ in range(total_cls + 1)]
	if(single_detection_each_class):
		for cls in range(1, total_cls + 1):
			if(len(overlap_threshed[cls]) == 0):
				continue
			max_confidence = np.argmax([detection_npy[index][2] for index in overlap_threshed[cls]])
			single_detection_threshed[cls].append(overlap_threshed[cls][max_confidence])
	else:
		for cls in range(1, total_cls + 1):
			single_detection_threshed[cls].extend(overlap_threshed[cls])


	#whether only keep the top N classes
	top_classes_threshed = [[] for _ in range(total_cls + 1)]
	max_confidence_dic = {}
	for cls in range(1, total_cls + 1):
		if(len(single_detection_threshed[cls]) == 0):
			continue
		max_confidence = np.max([detection_npy[index][2] for index in single_detection_threshed[cls]])
		max_confidence_dic[cls] = max_confidence
	if len(max_confidence_dic.keys()) <= max_number_of_cls:
		top_classes_threshed = single_detection_threshed
	else:
		inverse = [(value, key) for key, value in max_confidence_dic.items()]
		for i in range(max_number_of_cls):
			max_cls = max(inverse)[1]
			inverse.remove(max(inverse))
			top_classes_threshed[max_cls].extend(single_detection_threshed[max_cls])

	#combine all classes
	results = np.array([])
	count = 0
	for cls in range(1, total_cls + 1):
		if(len(top_classes_threshed[cls]) == 0):
			continue
		count = count + 1
		result = np.array([detection_npy[index] for index in top_classes_threshed[cls]])
		if results.shape[0] != 0:
			results = np.concatenate((results, result), axis=0)
		else:
			results = result
	if class_count_dic.has_key(count):
		class_count_dic[count] = class_count_dic[count] + 1
	else:
		class_count_dic[count] = 1

	# This step ensures the 10s clip used contains an optimal number of detected objects.
	# Might need to be tuned for different datasets.
	# It can be useful to deal with very noisy datasets like AudioSet. 
	if results.shape[0] >= 60 and results.shape[0] <= 300:
		np.save(os.path.join(top_detections_root, detection), results)
		print(count) #number of classes detected in the video

print(class_count_dic)