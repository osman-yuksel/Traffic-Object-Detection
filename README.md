# Traffic Object Detection
This program can detect daily life traffic objects from fed footage; cars, humans, traffic lights, etc. Created for educational purposes and not meant to be used in real-life cases. Use with your own discretion!

![image](https://github.com/osman-yuksel/Traffic-Object-Detection/blob/master/screenshot.jpg)

<h2>Requirements</h2>

1. Python 3.8.x (other versions might cause problems).
	1. Numpy
	2. CV2
	3. Tenserflow
	4. [Tensorflow-Utils](https://github.com/tensorflow/models/tree/master/research/object_detection/utils)
2. CUDA Toolkit.
3. NVIDIA cuDNN.
4. [ssd_resnet50_v1_fpn_640x640 Detection Model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
	1. You can use which model you prefer, just dont forget to update paths and labelmap.

<h2>How to use?</h2>

1. Add a video file in the same folder as the main.py file. 
2. Change the name of the video file to "input_video.mp4". This name is the program's default, which can be changed via path variables in main.py file. This change is necessary if you're using a video format other than mp4.
3. Now you can simply run the main.py script.
