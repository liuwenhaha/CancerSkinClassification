import nrrd
import numpy as np
import cv2
import os
import pandas as pd

root_dir = "./skin-cancer-mnist-ham10000"
metadata_path = "{}/HAM10000_metadata.csv".format(root_dir)
directory_to_write = "./nrrd"
df = pd.read_csv(metadata_path)[['image_id', 'dx']]
labels = df.dx.unique()
make_jpg = lambda x: "{}.jpg".format(x)
l = len(df)


def get_image_path(filename, root_dir):
    directories = os.listdir(root_dir)
    directory = None
    for i in directories:
        if "." not in i and filename in os.listdir("{}/{}".format(root_dir, i)):
            directory = i
    return "{}/{}/{}".format(root_dir, directory, filename)


def read_image_to_np_arr(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def write_to_nrdd(file_path, image_arr):
    label = np.ones(shape=image_arr.shape)
    nrrd.write("{}.nrrd".format(file_path), image_arr)
    nrrd.write("{}-label.nrrd".format(file_path), label)


for label in labels:
    try:
        os.mkdir("{}/{}".format(directory_to_write, label))
    except FileExistsError:
        pass

count = 0
for index, row in df.iterrows():
    diagnosis = row['dx']
    image_name = row['image_id']
    image_path = get_image_path(make_jpg(image_name), root_dir)
    image = read_image_to_np_arr(image_path)
    to_path = "{}/{}/{}".format(directory_to_write, diagnosis, image_name)
    write_to_nrdd(to_path, image)
    count += 1
    print(count / l)
