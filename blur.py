# File containing the actual blurring code

import cv2 as cv
import numpy as np
from PIL import Image

# Blurs an image via Box-blur method in a certain area
def blur(image, area):
	cropped = image.crop(area)

	small = cropped.resize((32, 32), resample=Image.BILINEAR)	
	cropped = small.resize(cropped.size, Image.NEAREST) 
	
	image.paste(cropped, area)

face_cascade = cv.CascadeClassifier()
plate_cascade = cv.CascadeClassifier()

# Load cascade models

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
		print(face)
		blur(actual_img, face)
		cv.rectangle(img, face, (0, 0, 0))

	actual_img.save("data/test1.jpg")
	cv.imshow("Display window", img)
	cv.waitKey(0)
