# File containing the actual blurring code

import cv2 as cv
import numpy as np
from PIL import Image
import moviepy.editor as mpe
import time
import threading
import os

def blur(image, area):
	cropped = image.crop((area[0], area[1], area[0] + area[2], area[1] + area[3]))

	# Create a copy for later use
	original_size = cropped

	# Size down and then resize up to blur the image
	small = cropped.resize((16, 16), resample=Image.BILINEAR)	
	cropped = small.resize(original_size.size, Image.NEAREST) 
	
	image.paste(cropped, (area[0], area[1]))


# Load cascade models
face_cascade = cv.CascadeClassifier()
plate_cascade = cv.CascadeClassifier()

if not face_cascade.load("data/haarcascade_frontalface_default.xml"):
	print("Error -- Loading face cascade")
	exit()

if not plate_cascade.load("data/haarcascade_russian_plate_number.xml"):
	print("Error -- Loading number-plate cascade")
	exit()

processed_faces = []
processed_plates = []


# simple functions for processing a chunk of images and returning the faces and numberplates
def process_face_chunk(frames):
	faces = []
	for frame in frames:
		faces = face_cascade.detectMultiScale(frame)
		processed_faces.append(faces)
		
def process_plate_chunk(frames):
	plates = []
	for frame in frames:
		plates = plate_cascade.detectMultiScale(frame)
		processed_plates.append(plates)
#


def blur_video(filename, want_faces, want_plates):
	start_time = time.time()

	# Filenames set up
	temp_filename = filename.split('.')[0] + "_temp.mp4"
	final_filename = filename.split('.')[0] + "_blurred.mp4"

	# Load video
	vid = cv.VideoCapture(filename)
	
	if not vid.isOpened():
		print("Error loading video")
		exit

	vid_width = vid.get(cv.CAP_PROP_FRAME_WIDTH)
	vid_height = vid.get(cv.CAP_PROP_FRAME_HEIGHT)
	frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
	fps = vid.get(cv.CAP_PROP_FPS)

	video_frames = []
	frames = []

	# Get and store all frames to process
	for i in range(frame_count):
		ret, frame = vid.read()

		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		frames.append(gray)
		video_frames.append(Image.fromarray(frame).convert('RGB'))

	# Split data into two chuncks
	HALF_LENGTH = int(len(frames) / 2)
	chunk1 = frames[:HALF_LENGTH]
	chunk2 = frames[HALF_LENGTH:len(frames) - HALF_LENGTH]

	# Process data if the user wants to
	if want_faces:
		face_thread1 = threading.Thread(target=process_face_chunk, args=(chunk1,))
	
	if want_plates:
		plate_thread1 = threading.Thread(target=process_plate_chunk, args=(chunk1,))

	if want_faces:
		face_thread1.start()

	if want_plates:
		plate_thread1.start()

	if want_faces:
		face_thread1.join()

	if want_plates:
		plate_thread1.join()

	face_thread2 = threading.Thread(target=process_face_chunk, args=(chunk2,))
	plate_thread2 = threading.Thread(target=process_plate_chunk, args=(chunk2,))

	face_thread2.start()
	plate_thread2.start()
	face_thread2.join()
	plate_thread2.join()
	print("Processing of images took %s"%(time.time() - start_time))
	start_time = time.time()

	# Blur all needed areas in each video frane
	for i in range(len(processed_faces)):
		for face in processed_faces[i]:
			blur(video_frames[i], face)

	for i in range(len(processed_plates)):
		for plate in processed_plates[i]:
			blur(video_frames[i], plate)

	print("Blurring of video frames took %s"%(time.time() - start_time))

	fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
	out = cv.VideoWriter(temp_filename, fourcc, fps, (int(vid_width), int(vid_height)), True)
	
	# Write out video frames
	for frame in video_frames:
		out.write(np.array(frame))

	vid.release()
	out.release()

	# Copy audio tracks across files
	original_vid = mpe.VideoFileClip(filename)
	original_vid.audio.write_audiofile("data/extracted.mp3")

	new_vid = mpe.VideoFileClip(temp_filename)
	audio = mpe.AudioFileClip("data/extracted.mp3")

	final = new_vid.set_audio(audio)

	final.write_videofile(final_filename)

	# Remove unessary files
	os.remove("data/extracted.mp3")
	os.remove(temp_filename)


def blur_image(filename, want_faces, want_plates):
	# Load the actual image
	img = cv.imread(filename, 0)

	# Get faces and plates from image
	faces = []
	number_plates = []

	if want_faces == "true":
		faces = face_cascade.detectMultiScale(img)
	
	if want_plates == "true":
		number_plates = plate_cascade.detectMultiScale(img)
	
	actual_img = Image.open(filename)

	# Blur all that data
	for face in faces:
		blur(actual_img, face)

	for plate in number_plates:
		blur(actual_img, plate)

	actual_img.save(filename)
