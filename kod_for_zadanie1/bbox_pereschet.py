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


# Обрезаем наш датасет из 15к изображений
dataset_file = os.path.join(base_path, "my_dataset_15000.txt")

# Загружаем файл с landmarks
landmarks_path = os.path.join(base_path, "list_landmarks_celeba.txt")
all_landmarks = pd.read_csv(
    landmarks_path, 
    sep='\s+',  
    skipinitialspace=True  
)

output_landmarks_file = os.path.join(base_path, "bbox_crop_landmarks.txt")

def final_crop(mas_names):
    with open(output_landmarks_file, 'w', encoding='utf-8') as f:
            header = "image_id,lefteye_x,lefteye_y,righteye_x,righteye_y,nose_x,nose_y,leftmouth_x,leftmouth_y,rightmouth_x,rightmouth_y"
            f.write(header + "\n")

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

        landmarks_row = all_landmarks[all_landmarks['image_id'] == img_name]
        landmarks_row = landmarks_row.iloc[0]

        original_coords = [
            landmarks_row['lefteye_x'], landmarks_row['lefteye_y'],
            landmarks_row['righteye_x'], landmarks_row['righteye_y'],
            landmarks_row['nose_x'], landmarks_row['nose_y'],
            landmarks_row['leftmouth_x'], landmarks_row['leftmouth_y'],
            landmarks_row['rightmouth_x'], landmarks_row['rightmouth_y']
        ]

        # Пересчитываем координаты (вычитаем смещение bbox)
        cropped_coords = []
        for i in range(0, len(original_coords), 2):
            x_cropped = original_coords[i] - x1
            y_cropped = original_coords[i+1] - y1
            cropped_coords.extend([x_cropped, y_cropped])
        
        with open(output_landmarks_file, 'a', encoding='utf-8') as f:
            coords_str = ",".join([f"{coord:.6f}" for coord in cropped_coords])
            f.write(f"{img_name},{coords_str}\n")

        img.close()
        cropped_img.close()

with open(dataset_file, 'r') as f:
    my_images = [line.strip() for line in f if line.strip()]

final_crop(my_images)