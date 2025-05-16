import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join

images_dir = "images"             # 圖檔目錄
annotations_dir = "annotations"   # VOC 標註檔目錄
output_dir = "images"             # YOLO 標註檔的輸出目錄
classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']

def getImagesInDir(dir_path, image_type=".png"):
    image_list = []
    for filename in glob.glob(dir_path + '\\*'+ image_type):
        image_list.append(filename)

    return image_list

def convert(size, box):
    if size[0] == 0:
        dw = 1./(size[0]+0.00001)
    else:
        dw = 1./(size[0])
        
    if size[0] == 0:
        dh = 1./(size[1]+0.00001)
    else:
        dh = 1./(size[1])

    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '\\' + basename_no_ext + '.xml')
    out_file = open(output_path + '\\' + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

cwd = getcwd()
full_images_path = cwd + '\\' + images_dir
full_annotations_path = cwd + '\\' + annotations_dir
output_path = cwd +'\\' + output_dir

image_paths = getImagesInDir(full_images_path)
list_file = open(full_images_path + '_list.txt', 'w')

for image_path in image_paths:
    print(image_path)
    list_file.write(image_path + '\n')
    convert_annotation(full_annotations_path, output_path, image_path)
list_file.close()

print("Finished processing: " + output_path)