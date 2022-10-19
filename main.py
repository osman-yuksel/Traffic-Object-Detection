# Requeired libraries
import tensorflow as tf
import numpy as np
import time
import cv2 
import os

from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from keras.models import load_model
from video import make_video



# For calulate total time of the detection runs for project
main_time = time.time()


# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


PY_PATH = os.path.dirname(os.path.realpath(__file__))

# PATHS
PATH_TO_MODEL_DIR = PY_PATH + "/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model"
PATH_TO_CLASSIFICATION_MODEL = PY_PATH + "/my_model.h5"
PATH_TO_LABELS = PY_PATH + "/mscoco_label_map.pbtxt"
PATH_TO_DETECTED = PY_PATH + "/detected/"
MP4_OUTPUT_PATH = PY_PATH + "/output_video.mp4"
MP4_INPUT_PATH = PY_PATH + "/input_video.mp4"




# Loading the detection model
print('Loading models...', end='')
start_time = time.time()

# Load saved models and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)
classification_model = load_model(PATH_TO_CLASSIFICATION_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


# Loading the label data
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)


###############################################################################################



def lanesDetection(img):

    # Masking data for the image
    region_of_interest_vertices = [
        
        #(250,1080),(780,700),(1250,700),(1920,1600) 
        (250,1080),(700,750),(780,750),(750,1080),(1250,1080),(1250,750),(1330,750),(1750,1080)   
    ]
    # Convert frame to grayscale 
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Convert grayscale image to outline borders of objects
    blur=cv2.GaussianBlur(gray_img,(5,5),0)
    edge = cv2.Canny(blur, 100, 200)

    # Crops the image to the desired region, not sure how exactly this works
    cropped_image = region_of_interest(
        edge, np.array([region_of_interest_vertices], np.int32))
    # Use OpenCVs HoughsLines line detection
    # OpenCVs HoughsLines çizgi belirleme kulanılır
    # See https://learnopencv.com/hough-transform-with-opencv-c-python/ for more Info
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180,
                           threshold=50, lines=np.array([]), minLineLength=10, maxLineGap=30)
    
    # Draw the detected Lines into the original frame
    if lines is None:
        cv2.putText(img,'Serit yok ', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,250),2,2)
        
        return img
    else:
      image_with_lines = draw_lines(img, lines)
      return image_with_lines


# Masking the image 
def region_of_interest(img, vertices):

    mask = np.zeros_like(img)
    match_mask_color = (255)
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Drawing lines
def draw_lines(img, lines):
    # Creates a copy of the original image
    img = np.copy(img)
    # Creates a blank image with the dimensions of of the original image
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    control=1
    
    # Draw the found Lanes as lines onto the blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
             #Calculating if it is line
            if (((y1-y2)/(x1-x2))>0.4) or (((y1-y2)/(x1-x2))<(-0.4)):
               cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
             continue
             #Checking for if the car is in line
            if ((500<x1<700 or 500<x2<700) or(1300<x1<700 or 1500<x2<700)) and (800<y1<1000 or 800<y2<1000):
                control=0

    #Printing to the image
    if control==0:
         cv2.putText(img,'Serit Korunuyor ', (700,200), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,250,0),2,1)
    elif control==1:
         cv2.putText(img,'Seriten Cikiliyor ', (700,200), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,250),2,1)
    else:
         cv2.putText(img,'HATA', (700,360), cv2.FONT_HERSHEY_SIMPLEX, 5,(0,0,250),5,5)                

    # Create an image that overlays the blank image with lines onto the original frame
    img = cv2.addWeighted(img, 1, blank_image, 1, 0.0)
    img = cv2.rectangle(img, pt1=(500,800), pt2=(700,1000), color=(255,0,0), thickness=1)
    img = cv2.rectangle(img, pt1=(1300,800), pt2=(1500,1000), color=(255,0,0), thickness=1)
    return img



###############################################################################################



def find_color(img_cord, light_frame):

    y_min_F = int(img_cord[0].item() * 1080)
    y_max_F = int(img_cord[2].item() * 1080)
    x_min_F = int(img_cord[1].item() * 1920)
    x_max_F = int(img_cord[3].item() * 1920)

    #print(type(y_min_F), y_min_F)
    #print(type(y_max_F), y_max_F)
    #print(type(x_min_F), x_min_F)
    #print(type(x_max_F), x_max_F)

    #print(light_frame.shape)
    img_map = light_frame[y_min_F:y_max_F , x_min_F:x_max_F]
    #cv2.imshow("ww",img_map)
    #cv2.waitKey(0)
    
    img_map = cv2.resize(img_map, (50,50))
    #img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_map, 0)

    predictions = classification_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ['green', 'red', 'yellow']

    color = class_names[np.argmax(score)], 100*np.max(score)


    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # org
    org = (x_min_F, y_max_F + 20)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    if color[0] == "red":
        colors = (255,0,0)
    if color[0] == "green":
        colors = (0,255,0)
    if color[0] == "yellow":
        colors = (255,255,0)
    # Line thickness of 2 px
    thickness = 2
   
    # Using cv2.putText() method
    light_frame = cv2.putText(light_frame, str(color[0]), org, font, fontScale, 
                 colors, thickness, cv2.LINE_AA, False)


    return light_frame


###############################################################################################



def object_detection(path_to_detection_image, i):
    IMAGE_PATH = path_to_detection_image
    #test_image = Image.open(IMAGE_PATH)

    # Changing the image to nparray
    def load_image_into_numpy_array(path):
        return np.array(Image.open(path))

    # Starting the detection
    print('Detecting...', end='')
    start_time = time.time()

    
    
    image_path = IMAGE_PATH
    #print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)



    # Changing np array to tensor
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Running interface
    detections = detect_fn(input_tensor)



    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    print(detections.keys())


    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()


    # Total detected objects
    total_detection = len(detections['detection_boxes'])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))



    # Deleting unnecessary detections
    print("Total detection before deletion: ", total_detection)

    y_min = 0.1
    x_min = 0.1
    y_max = 0.9
    x_max = 0.9

    deletion_index = 0
    traffic_light_cord = np.array([(0,0,0,0)])

    for x in range(total_detection):
    #print(x)
        if detections['detection_scores'][x-deletion_index] > 0.3:
            if detections['detection_boxes'][x-deletion_index][0] < y_min or detections['detection_boxes'][x-deletion_index][1] < x_min or detections['detection_boxes'][x-deletion_index][2] > y_max or detections['detection_boxes'][x-deletion_index][3] > x_max:
                detections['detection_boxes'] = np.delete(detections['detection_boxes'], x-deletion_index, axis=0)
                detections['detection_classes'] = np.delete(detections['detection_classes'], x-deletion_index)
                detections['detection_scores'] = np.delete(detections['detection_scores'], x-deletion_index)
                deletion_index += 1 
            elif detections['detection_classes'][x-deletion_index] == 10:
                traffic_light_cord = np.vstack((traffic_light_cord, detections['detection_boxes'][x-deletion_index]))
        else:
            detections['detection_boxes'] = np.delete(detections['detection_boxes'], x-deletion_index, axis=0)
            detections['detection_classes'] = np.delete(detections['detection_classes'], x-deletion_index)
            detections['detection_scores'] = np.delete(detections['detection_scores'], x-deletion_index)
            deletion_index += 1


    total_detection = len(detections['detection_boxes'])
    print("Total detection after deletion: ", total_detection)

    # Line detection    
    image_np_with_detections = lanesDetection(image_np)


    #Traffic light color detection
    rows = len(traffic_light_cord)
    for row in range(rows):
        if not row == 0:
            image_np_with_detections = find_color(traffic_light_cord[row], image_np_with_detections)



    # Visualizing detections
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.0,
            agnostic_mode=False)

    #print(type(image_np_with_detections))
    # Showing image
    img = Image.fromarray(image_np_with_detections, 'RGB')
    img.save(PATH_TO_DETECTED + 'detected_frame' + str(i) + '.jpg')


###############################################################################################



# Main function
# Video source location
cap = cv2.VideoCapture(MP4_INPUT_PATH)

counter = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    try:
        cv2.imwrite(os.path.join(PATH_TO_DETECTED, "FRAME.jpg"), frame)
    except:
        print("END OF THE VIDEO FILE")
        break
    object_detection(os.path.join(PATH_TO_DETECTED, 'FRAME.jpg'), counter)
    print(counter)
    cv2.imshow("Video", cv2.resize(cv2.imread(os.path.join(PATH_TO_DETECTED, 'detected_frame' + str(counter) + '.jpg')), (1280, 720)))
    cv2.waitKey(1)
    counter += 1
    
    main_end_time = time.time()
    print("Total time: ", main_end_time - main_time)

cap.release()
cv2.destroyAllWindows()

make_video(PATH_TO_DETECTED, MP4_OUTPUT_PATH)




