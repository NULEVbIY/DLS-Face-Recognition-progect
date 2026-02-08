import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from PIL import Image
import glob

# Пути
base_path = r"C:\Users\Public\Documents\Face recognition"
work_landmarks_path = os.path.join(base_path, "work_dataset_landmarks.txt")
landmarks_df = pd.read_csv(work_landmarks_path, sep='\s+')

def create_heatmap(size, landmark, sigma=2):
    x, y = landmark
    h, w = size
    
    x = min(max(0, int(x)), w - 1)
    y = min(max(0, int(y)), h - 1)
    
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    heatmap = np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * sigma**2))
    return heatmap

def landmarks_to_heatmaps(image_shape, landmarks, sigma=2):
    heatmaps = []
    for (x, y) in landmarks:
        hm = create_heatmap(image_shape, (x, y), sigma=sigma)
        heatmaps.append(hm)
    return np.array(heatmaps)

def create_heatmap_for_image(img_name, sigma=2, image_size=(112, 112)):
    row = landmarks_df[landmarks_df['image_id'] == img_name]
    
    if row.empty:
        return None
    
    row = row.iloc[0]
    
    landmarks = [
        (row['lefteye_x'], row['lefteye_y']),
        (row['righteye_x'], row['righteye_y']),
        (row['nose_x'], row['nose_y']),
        (row['leftmouth_x'], row['leftmouth_y']),
        (row['rightmouth_x'], row['rightmouth_y'])
    ]
    
    heatmaps = landmarks_to_heatmaps(image_size, landmarks, sigma=sigma)
    return heatmaps

def visualize_heatmaps(img_name, sigma=2):
    img_path = os.path.join(base_path, "Work_Dataset", img_name)
    img = Image.open(img_path)
    
    heatmaps = create_heatmap_for_image(img_name, sigma=sigma)
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Изображение')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(img)
    row = landmarks_df[landmarks_df['image_id'] == img_name].iloc[0]
    landmarks = [
        (row['lefteye_x'], row['lefteye_y']),
        (row['righteye_x'], row['righteye_y']),
        (row['nose_x'], row['nose_y']),
        (row['leftmouth_x'], row['leftmouth_y']),
        (row['rightmouth_x'], row['rightmouth_y'])
    ]
    
    for i, (x, y) in enumerate(landmarks):
        axes[1, 0].plot(x, y, 'ro', markersize=4)
    axes[1, 0].set_title('Точки')
    axes[1, 0].axis('off')
    
    landmark_names = ['Левый глаз', 'Правый глаз', 'Нос', 'Левый рот', 'Правый рот']
    
    for i in range(5):
        row_idx = i // 3  # 0, 0, 0, 1, 1
        col_idx = (i % 3) + 1  # 1, 2, 3, 1, 2
        
        # Heatmap
        axes[row_idx, col_idx].imshow(heatmaps[i], cmap='hot')
        axes[row_idx, col_idx].set_title(landmark_names[i])
        axes[row_idx, col_idx].axis('off')
        
        # Наложенный heatmap (колонки 4, 5, 0, 4, 5)
        overlay_col = col_idx + 3 if col_idx < 3 else col_idx - 3
        axes[row_idx, overlay_col].imshow(img, alpha=0.7)
        axes[row_idx, overlay_col].imshow(heatmaps[i], cmap='hot', alpha=0.5)
        axes[row_idx, overlay_col].set_title(f'{landmark_names[i]}')
        axes[row_idx, overlay_col].axis('off')
    
    plt.suptitle(f'{img_name}')
    plt.tight_layout()
    plt.show()
    
    img.close()
    
    return heatmaps

def create_all_heatmaps_dataset(save_dir=None, sigma=2, image_size=(112, 112)):
    if save_dir is None:
        save_dir = os.path.join(base_path, "Heatmaps_Dataset")
    
    os.makedirs(save_dir, exist_ok=True)
    
    heatmaps_data = []
    
    for idx, row in landmarks_df.iterrows():
        img_name = row['image_id']
        
        landmarks = [
            (row['lefteye_x'], row['lefteye_y']),
            (row['righteye_x'], row['righteye_y']),
            (row['nose_x'], row['nose_y']),
            (row['leftmouth_x'], row['leftmouth_y']),
            (row['rightmouth_x'], row['rightmouth_y'])
        ]
        
        heatmaps = landmarks_to_heatmaps(image_size, landmarks, sigma=sigma)
        
        heatmap_file = os.path.join(save_dir, f"{img_name.replace('.jpg', '.npy')}")
        np.save(heatmap_file, heatmaps)
        
        heatmaps_data.append({
            'image_id': img_name,
            'heatmap_file': f"{img_name.replace('.jpg', '.npy')}"
        })
    
    metadata_df = pd.DataFrame(heatmaps_data)
    metadata_path = os.path.join(save_dir, "heatmaps_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"Сохранено: {save_dir}")
    print(f"Файлов: {len(heatmaps_data)}")
    
    return save_dir

heatmaps_dir = create_all_heatmaps_dataset(sigma=2)