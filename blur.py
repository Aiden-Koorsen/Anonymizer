# File containing the actual blurring code

import cv2 as cv
import numpy as np
from PIL import Image

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


def blur_video(filename):
	# Load video
	vid = cv.VideoCapture(filename)
	
	if not vid.isOpened():
		print("Error loading video")
		exit

	vid_width = vid.get(cv.CAP_PROP_FRAME_WIDTH)
	vid_height = vid.get(cv.CAP_PROP_FRAME_HEIGHT)
	
	# Define codec and writer to change video
	fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
	out = cv.VideoWriter(filename.split('.')[0] + "_new.mp4", fourcc, 30.0, (int(vid_width), int(vid_height)), True)

	# Loop until video is finished
	while True:
		ret, frame = vid.read()
		
		if not ret:
			print("No more frames: Exiting")
			break

		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		
		# Get faces and plates from frame
		faces = face_cascade.detectMultiScale(gray)
		number_plates = plate_cascade.detectMultiScale(gray)
		
		actual_img = Image.fromarray(frame).convert('RGB')

		# Blur
		for face in faces:
			blur(actual_img, face)

		for plate in number_plates:
			blur(actual_img, plate)			
	
		new_frame = np.array(actual_img)

		# Write out to new file
		out.write(new_frame)
		
	# Release the video data	
	vid.release()
	out.release()
		

def blur_image(filename):
	# Load the actual image
	img = cv.imread(filename, 0)

	# Get faces and plates from image
	faces = face_cascade.detectMultiScale(img)
	number_plates = plate_cascade.detectMultiScale(img)
	
	actual_img = Image.open(filename)

	# Blur all that data
	for face in faces:
		blur(actual_img, face)

	for plate in number_plates:
		blur(actual_img, plate)

	actual_img.save(filename)
