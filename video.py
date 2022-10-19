import cv2
import glob




def make_video(png_path, output_path):
    frameSize = (1920, 1080)
    framerate = 30
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
    print("Creating video file... ", frameSize, framerate)

    out = cv2.VideoWriter(output_path,fourcc, framerate, frameSize)

    for filename in sorted(glob.glob(png_path + "*.jpg"), key=len):
        print(filename)
        img = cv2.imread(filename)
        out.write(img)

    out.release()
    print("Done!")
