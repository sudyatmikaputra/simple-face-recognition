import face_recognition
import os
import pickle
import cv2
import time
from threading import Thread

KNOWN_FACES_DIR = "known_faces"
# UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"
fps = 1/8
fpssec = fps *1000
video = cv2.VideoCapture(0)

# video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
# fps = 1/30
# fps_ms = int(fps*1000)


print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
		image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
		encoding = face_recognition.face_encodings(image)[0]
		known_faces.append(encoding)
		known_names.append(name)

print("processing unknown faces")
while True:
	ret, image = video.read()
	res = cv2.resize(image, (270, 270)) 

	locations = face_recognition.face_locations(res, model=MODEL)
	encodings = face_recognition.face_encodings(res, locations)

	for face_encoding, face_location in zip(encodings, locations):
		results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
		match = None
		if True in results:
			match = known_names[results.index(True)]
			print(f"Match found: {match}")
			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1], face_location[2])
			color = [0, 255, 0]
			cv2.rectangle(res, top_left, bottom_right, color, FRAME_THICKNESS)

			top_left = (face_location[3], face_location[2])
			bottom_right = (face_location[1], face_location[2]+22)

	cv2.imshow("haha",res)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break