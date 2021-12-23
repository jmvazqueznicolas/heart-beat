import cv2
import pyramids
import heartrate
import preprocessing
import eulerian
from tqdm import tqdm
import time

# Frequency range for Fast-Fourier Transform
freq_min = 1
freq_max = 1.8

# Preprocessing phase
print("Recopilando imagenes")
video_frames, frame_ct, fps = preprocessing.read_video("videos/marlon.webm")


print("Procesando información")

lap_video = pyramids.build_video_pyramid(video_frames)

amplified_frames = []

for i, video in enumerate(tqdm(lap_video)):
    if i == 0 or i == len(lap_video)-1:
        continue

    result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
    lap_video[i] += result

    heart_rate = int(heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max))

amplified_frames = pyramids.collapse_laplacian_video_pyramid(lap_video, frame_ct)

# Output heart rate and final video
print("Frecuencia cardiaca: ", heart_rate, "latidos por minuto")
time.sleep(3)
#for frame in amplified_frames:
#    cv2.imshow("frame", frame)
#    cv2.waitKey(1)


