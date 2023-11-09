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

if not plate_cascade.load("data/haarcascade_license_plate_rus_16stages.xml"):
	print("Error -- Loading number-plate cascade")
	exit()


def blur_video(filename):
	...

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

	actual_img.save("data/test1.jpg")
