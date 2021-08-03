import os
import sys
import random
import shutil

path = sys.argv[1]
num_partition = int(sys.argv[2])

image = os.path.join(path, "coco128/images/train2017")
label = os.path.join(path, "coco128/labels/train2017")

# walk through all data
images = []
labels = []
for _, dir_list, file_list in os.walk(image):
    for file_name in file_list:
        if ".jpg" in file_name:
            images.append(os.path.join(image, file_name))
            labels.append(os.path.join(label, file_name.replace(".jpg", ".txt")))

index = list(range(len(images)))
num_items = int(len(images)/num_partition)

for i in range(num_partition):
    sub_path = os.path.join(path, str(i+1))
    image_i = os.path.join(sub_path, "coco128/images/train2017")
    label_i = os.path.join(sub_path, "coco128/labels/train2017")
    if os.path.exists(image_i) == False:
        os.makedirs(image_i)
    if os.path.exists(label_i) == False:
        os.makedirs(label_i)
    # draw num_items from index
    slice = random.sample(index, num_items)
    # print(slice)
    # save to dir
    for ii in slice:
        shutil.copy2(images[ii], image_i)
        shutil.copy2(labels[ii], label_i)
    