import requests
import nltk 
import random
import warnings
import matplotlib.pyplot as plt
from IPython.display import Image
import numpy as np
import matplotlib.image as mpimg
import cv2
warnings.filterwarnings("ignore")

url = 'http://www.gutenberg.org/files/28054/28054-0.txt'
r = requests.get(url)
text_file = open("Output.txt", "w")
text_file.write(r.text)
text_file.close()

f=open('Output.txt')
raw=f.read()
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)

def image_generator():
	width = 400
	height = 600
	img = np.zeros([width, height, 3], dtype=np.uint8)
	img.fill(255)
    empty = np.zeros([width, height, 3], dtype=np.uint8)
    empty.fill(255)
	font_thickness_list = [0.5,0.7,0.8]
	fonts_list = [cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX , cv2.FONT_ITALIC]
	font_thickness = random.choice(font_thickness_list)
	font = random.choice(fonts_list)
	begin = random.randint(0,len(text)-140)
	end = begin+140
	l = 20
	for i in range(20):
		vstavka = ' '.join(text[begin:end])
		vstavka = vstavka.replace(" .", ".").replace(" ,", ",").replace("???", '').replace("?","")
		begin = begin + 7
		end = end + 7
		cv2.putText(img,vstavka, (0,l),font,font_thickness,10)
		l = l+20


	pts_src = np.array([[0, 0], [600, 0], [600, 400],[0, 400]])
	pts_dst = np.array([[random.uniform(0,120), random.uniform(0,80)],[random.uniform(480,600), random.uniform(0,80)],[random.uniform(480,600),random.uniform(320,400)],[random.uniform(0,120), random.uniform(320,400)]])
	
	# Calculate Homography
	h, status = cv2.findHomography(pts_src, pts_dst)
	 
	# Warp source image to destination based on homography
	im_out = cv2.warpPerspective(img, h, (img.shape[1],img.shape[0]))
    im_out = im_out + empty
	#plt.imshow(im_out)
	return im_out, pts_dst

def dataset_generator(num_images):
	image_list = []
	angle_points = []
	for i in range(num_images):
		image = image_generator()
		image_list.append(image[0])
		angle_points.append(image[1].reshape(8,1))
	return image_list,angle_points