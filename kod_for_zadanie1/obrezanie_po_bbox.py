import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# Базовый путь
base_path = r"C:\Users\Public\Documents\Face recognition"

# Загружаем bbox данные
bbox_path = os.path.join(base_path, "list_bbox_celeba.csv")
all_bbox = pd.read_csv(bbox_path)

def smotr_po_odnomy(mas_names, n):
    for img_name in random.sample(mas_names, n):
        img_path = os.path.join(base_path, "All_orig_images", img_name)
        img = Image.open(img_path)

        # получаем координаты bbox для конкретного изображения и извлекаем их
        bbox_row = all_bbox[all_bbox['image_id'] == img_name].iloc[0]
        x1 = bbox_row['x_1']
        y1 = bbox_row['y_1']
        width = bbox_row['width']
        height = bbox_row['height']

        # отрисовываем bbox на изображение
        plt.figure(figsize=(8, 6))
        plt.imshow(img)

        bbox_rectangle = patches.Rectangle(
            (x1, y1),      
            width, height,     
            linewidth=3,       
            edgecolor='green',  
            facecolor='none'   
        )
        plt.gca().add_patch(bbox_rectangle)
        plt.show()


def crop_face(img_name):
    img_path = os.path.join(base_path, "All_orig_images", img_name)
    img = Image.open(img_path)
    
    # Получаем bbox
    bbox_row = all_bbox[all_bbox['image_id'] == img_name].iloc[0]
    x1 = bbox_row['x_1']
    y1 = bbox_row['y_1']
    width = bbox_row['width']
    height = bbox_row['height']
    
    cropped_img = img.crop((x1, y1, x1 + width, y1 + height))
    return cropped_img

def show_cropped_face(img_name):
    cropped_img = crop_face(img_name)
    plt.figure(figsize=(8, 6))
    plt.imshow(cropped_img)
    plt.axis('off')
    plt.show()
    return cropped_img


test_images = ['140235.jpg', '000002.jpg']

# 1. Показываем bbox на оригинальных изображениях
# print("1. Показываем BBox на изображениях:")
smotr_po_odnomy(test_images, 2)

#-------------------------------------------------------------------------

# Обрезаем наш датасет из 15к изображений
# dataset_file = os.path.join(base_path, "my_dataset_15000.txt")

def final_crop(mas_names):
    for img_name in mas_names:
        img_path = os.path.join(base_path, "All_orig_images", img_name)
        img = Image.open(img_path)

        # получаем координаты bbox для конкретного изображения и извлекаем их
        bbox_row = all_bbox[all_bbox['image_id'] == img_name].iloc[0]
        x1 = bbox_row['x_1']
        y1 = bbox_row['y_1']
        width = bbox_row['width']
        height = bbox_row['height']

        cropped_img = img.crop((x1, y1, x1 + width, y1 + height))

        output_path = os.path.join(base_path, "My_cropped_images", img_name)
        cropped_img.save(output_path, 'JPEG', quality=95)

# with open(dataset_file, 'r') as f:
#     my_images = [line.strip() for line in f if line.strip()]

# final_crop(my_images)