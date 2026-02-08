import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import glob

# Базовый путь
base_path = r"C:\Users\Public\Documents\Face recognition"

# Загружаем ключевые точки
list_landmarks_path = os.path.join(base_path, "list_landmarks_celeba.txt")
all_landmarks = pd.read_csv(list_landmarks_path)

# Фото после обрезки по bbox
all_jpgs = glob.glob(os.path.join(base_path, "My_cropped_images", "*.jpg"))

def bicubic_resize_two(img_names):
    for img_name in img_names:
        img_path = os.path.join(base_path, "My_cropped_images", img_name)
        img = Image.open(img_path)
        
        resized_img = img.resize((112, 112), Image.Resampling.BICUBIC)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[1].imshow(resized_img)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        img.close()
        resized_img.close()
            
        
# Использование
# bicubic_resize_two(['140235.jpg', '031386.jpg'])

# Загружаем ключевые точки для обрезанных изображений
bbox_crop_landmarks_path = os.path.join(base_path, "bbox_crop_landmarks.txt")
cropped_landmarks = pd.read_csv(bbox_crop_landmarks_path)

# Создаем папку для результатов
output_dir = os.path.join(base_path, "Work_Dataset")
os.makedirs(output_dir, exist_ok=True)

# Размеры для ресайза
TARGET_SIZE = 112

# Список для хранения новых данных
new_landmarks_data = []

# Обрабатываем все изображения
for img_path in all_jpgs:
    img_name = os.path.basename(img_path)
    
    img = Image.open(img_path)
    orig_width, orig_height = img.size
    
    resized_img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BICUBIC)
    
    output_path = os.path.join(output_dir, img_name)
    resized_img.save(output_path, quality=95)
    
    # Получаем ключевые точки для этого изображения
    landmarks_row = cropped_landmarks[cropped_landmarks['image_id'] == img_name]
    
    if not landmarks_row.empty:
        landmarks_row = landmarks_row.iloc[0]
        
        # Извлекаем координаты
        orig_points = [
            (landmarks_row['lefteye_x'], landmarks_row['lefteye_y']),
            (landmarks_row['righteye_x'], landmarks_row['righteye_y']),
            (landmarks_row['nose_x'], landmarks_row['nose_y']),
            (landmarks_row['leftmouth_x'], landmarks_row['leftmouth_y']),
            (landmarks_row['rightmouth_x'], landmarks_row['rightmouth_y'])
        ]
        
        scale_x = TARGET_SIZE / orig_width
        scale_y = TARGET_SIZE / orig_height
        
        # Пересчитываем координаты
        new_points = []
        for x, y in orig_points:
            new_x = x * scale_x
            new_y = y * scale_y
            new_points.append((new_x, new_y))
        
        new_landmarks_data.append({
            'image_id': img_name,
            'lefteye_x': new_points[0][0],
            'lefteye_y': new_points[0][1],
            'righteye_x': new_points[1][0],
            'righteye_y': new_points[1][1],
            'nose_x': new_points[2][0],
            'nose_y': new_points[2][1],
            'leftmouth_x': new_points[3][0],
            'leftmouth_y': new_points[3][1],
            'rightmouth_x': new_points[4][0],
            'rightmouth_y': new_points[4][1]
        })
    
    img.close()
    resized_img.close()

# Сохраняем новые ключевые точки
new_landmarks_df = pd.DataFrame(new_landmarks_data)

output_landmarks_path = os.path.join(base_path, "work_dataset_landmarks.txt")
new_landmarks_df.to_csv(output_landmarks_path, sep=' ', index=False)