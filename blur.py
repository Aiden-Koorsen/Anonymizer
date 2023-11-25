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
	current_frame = 0
	for frame in frames:
		if want_faces:
			faces = face_cascade.detectMultiScale(frame)
			for face in faces:
				blur(video_frames[current_frame], face)
		if want_plates:
			plates = plate_cascade.detectMultiScale(frame)
			for plate in plates:
					blur(video_frames[current_frame], plate)

		current_frame = current_frame + 1

	print(f"Blurring of video frames took {(time.time() - start_time)} seconds")

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
