import cv2
import numpy as np
from tqdm import tqdm

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")
color = (255, 0, 0)
thickness = 2

# Read in and simultaneously preprocess video
def read_video(path):
    cap = cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = []
    face_rects = ()
    # Se capturan 300 imagenes
     
    for i in tqdm(range(400)):
        while True:
            ret, img = cap.read()
            if not ret:
                break
            h, w = img.shape[:2]
            start_point = (int(w/2 - w*0.2), int(h/2 - h*0.3))
            end_point = (int(w/2 + w*0.2), int(h/2 + h*0.3))
            img_rect = img.copy()
            cv2.rectangle(img_rect, start_point, end_point, color, thickness)
            cv2.namedWindow('Image Camera')        # Create a named window
            cv2.moveWindow('Image Camera', 640,510)  # Move it to (40,30)
            cv2.imshow('Image Camera', img_rect)
            cv2.waitKey(1)

            # Buscar rostros solamente en el rectangulo
            img = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            roi_frame = img

            # Detect face
            #if len(video_frames) == 0:
        
            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)
            if len(face_rects) > 0:
                break

        # Select ROI
        #if len(face_rects) > 0:
        for (x, y, w, h) in face_rects:
            roi_frame = img[y:y + h, x:x + w]
        if roi_frame.size != img.size:
            #cv2.imshow('Rostros', roi_frame)
            #cv2.waitKey(1)
            roi_frame = cv2.resize(roi_frame, (100, 100))
            #roi_2show = roi_frame.copy()
            
            frame = np.ndarray(shape=roi_frame.shape, dtype="float")
            frame[:] = roi_frame * (1. / 255)
            video_frames.append(frame)

    frame_ct = len(video_frames)
    cap.release()

    return video_frames, frame_ct, fps
