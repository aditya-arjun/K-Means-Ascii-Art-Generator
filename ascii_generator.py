from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage import io
from skimage.transform import rescale, resize
import numpy as np
import random


def find_clusters(resized_image, num_clusters):
    image_height, image_width, image_depth = resized_image.shape
    resized_image = resized_image.reshape(image_height * image_width, image_depth)
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(resized_image)
    labels = labels.reshape(image_height, image_width)

    return labels

def generate_ascii(filename, num_chars, out_image_height, force_even):
    if force_even and out_image_height % 2 == 1:
        out_image_height += 1

    min_char = 32
    max_char = 127
    
    indices = random.sample(range(min_char, max_char), num_chars)

    char_replace = "".join([chr(index) for index in indices])

    image = io.imread(filename)
    orig_image_height, orig_image_width = image.shape[0], image.shape[1]

    scale_factor = out_image_height / orig_image_height

    # Constant Built to Scale Width to correct amount given character size
    width_height_multiplier = 2.1132075
    out_image_width = int(scale_factor * orig_image_width * width_height_multiplier)
    if force_even and out_image_width % 2 == 1:
        out_image_width += 1

    resized_image = resize(image, (out_image_height, out_image_width))

    labels = find_clusters(resized_image, num_chars)

    ascii_picture = []

    for i in range(out_image_height):
        out_string = "".join([char_replace[labels[i][j]] for j in range(out_image_width)])
        ascii_picture.append(out_string)

    return ascii_picture

def write_to_file(ascii_picture, outfilename, print_dim):
    f = open(outfilename, "w")
    if print_dim:
        f.write(f'{len(ascii_picture)} {len(ascii_picture[0])}\n')

    for i in range(len(ascii_picture)):
        f.write(ascii_picture[i] + "\n")

    f.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('-nc', '--num-chars', type=int, default=15, help='number of characters used in ASCII')
    parser.add_argument('-oh', '--out-height', type=int, default=100, help='height of ascii file in characters')
    parser.add_argument('-o', '--output', default="out.txt", help='output file for ascii art, default is out.txt')
    parser.add_argument('--print-dim', action='store_true', help='output picture dimensions')
    parser.add_argument('--force-even', action='store_true', help='force output dimension to be even')
    
    args = parser.parse_args()
    picture = generate_ascii(args.input, args.num_chars, args.out_height, args.force_even)
    write_to_file(picture, args.output, args.print_dim)
