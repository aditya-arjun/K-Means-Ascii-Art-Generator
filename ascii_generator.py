from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage import io
from scipy.misc import imresize
import numpy as np
import random


def find_clusters(resized_image, num_clusters):
	image_height, image_width, image_depth = resized_image.shape
	resized_image = resized_image.reshape(image_height * image_width, image_depth)
	kmeans = KMeans(n_clusters= num_clusters)
	labels = kmeans.fit_predict(resized_image)
	labels = labels.reshape(image_height, image_width)

	return labels

def generate_ascii(filename, num_chars = 20, out_image_height = 64):
	min_char = 32
	max_char = 127
	
	indexes = random.sample(range(min_char, max_char), num_chars)
	char_replace = ""
	for i in range(num_chars):
		char_replace += chr(indexes[i])

	image = io.imread(filename)
	orig_image_height, orig_image_width = image.shape[0], image.shape[1]

	scale_factor = out_image_height / orig_image_height
	width_height_multiplier = 2.1132075
	out_image_width = int(scale_factor * orig_image_width * width_height_multiplier)

	resized_image = imresize(image, size=(out_image_height, out_image_width))

	labels = find_clusters(resized_image, num_chars)

	ascii_picture = []

	for i in range(out_image_height):
		ascii_picture.append("")
		for j in range(out_image_width):
			ascii_picture[i] += char_replace[labels[i][j]]

	return ascii_picture

def write_to_file(ascii_picture, outfilename):
	file = open(outfilename, "w")
	for i in range(len(ascii_picture)):
		file.write(ascii_picture[i] + "\n")

	file.close()

picture = generate_ascii("negative_group.png", 20, 200)
write_to_file(picture, "negative_group_ascii.txt")
