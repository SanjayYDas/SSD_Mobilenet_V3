import cv2
import time
import numpy as np




configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'





model = cv2.dnn_DetectionModel(weightsPath, configPath)


classLabels = []
classFile = 'Labels.txt'
with open(classFile,'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')




model.setInputSize(416,416)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


cap = cv2.VideoCapture('test.mp4')

# Check if the video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open video")

# Initialize variables
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
pre_timeframe = 0
pre_timeframe = time.time()

#To Download the vedio
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output.mp4', fourcc,25,(1280,720))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280,720))
    new_timeframe = time.time()
    fps = 1 / (new_timeframe - pre_timeframe)
    pre_timeframe = new_timeframe

    starting_inference_time = time.time()
    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)
    ending_inference_time = time.time()
    inference_time = ending_inference_time - starting_inference_time
    inference_time = inference_time*1000

    if len(classIndex) != 0:
        for classIndex, confidence, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if classIndex <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[classIndex-1], (boxes[0]+20, boxes[1]+80), font, fontScale=font_scale, color=(0, 255, 255), thickness=3)
                cv2.putText(frame, str(round(confidence * 100, 1)), (boxes[0] + 50, boxes[1] + 30),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

    cv2.putText(frame, f'Inference Time: {inference_time:.3f} ms', (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,0,255), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 120), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Object Detection Tutorial", frame)
    output_video.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
output_video.release()
cv2.destroyAllWindows()




print("Inference Time:",inference_time)
print("FPS:",fps)
