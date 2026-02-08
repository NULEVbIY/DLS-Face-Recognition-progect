import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)

# Hourglass Module
class HourglassModule(nn.Module):
    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.downsample = nn.MaxPool2d(2, 2)

        if depth > 1:
            self.hg = HourglassModule(depth - 1, channels)
            self.low_res = ResidualBlock(channels, channels)
        else:
            self.hg = ResidualBlock(channels, channels)

        self.res_before = ResidualBlock(channels, channels)
        self.res_after = ResidualBlock(channels, channels)

    def forward(self, x):
        skip = self.res_before(x)
        h, w = x.shape[2:]

        low = self.downsample(x)
        low = self.hg(low)
        up = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)

        out = up + skip
        out = self.res_after(out)
        return out

# Heatmap Head
class HeatmapHead(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, num_keypoints, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.conv2(x) 

class StackedHourglassNetwork(nn.Module):
    def __init__(self, num_stacks=2, num_blocks=3, in_channels=3, out_channels=5,
                 channels=128, input_size=112):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_keypoints = out_channels
        self.channels = channels

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, channels//4, 7, stride=1, padding=3),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            ResidualBlock(channels//4, channels//2),
            nn.MaxPool2d(2, 2),
            ResidualBlock(channels//2, channels)
        )

        self.hourglasses = nn.ModuleList([HourglassModule(num_blocks, channels) for _ in range(num_stacks)])
        self.heads = nn.ModuleList([HeatmapHead(channels, out_channels) for _ in range(num_stacks)])

        self.intermediate_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for i in range(num_stacks-1)
        ])
        self.final_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for i in range(num_stacks-1)
        ])

    def forward(self, x):
        x = self.initial(x) 
        outputs = []

        for i in range(self.num_stacks):
            hg_out = self.hourglasses[i](x)  
            heatmap = self.heads[i](hg_out) 
            outputs.append(heatmap)

            if i < self.num_stacks - 1:
                features = self.intermediate_convs[i](hg_out)  
                heatmap_features = self.final_convs[i](heatmap)  
                x = x + features + heatmap_features 

        final_outputs = []
        for heatmap in outputs:
            final_heatmap = F.interpolate(heatmap, size=(112, 112), mode='bilinear', align_corners=False)
            final_outputs.append(final_heatmap)

        return final_outputs


device = torch.device('cpu')
print(f"Using device: {device}")

# Пути к файлам
base_path = r"C:\Users\Public\Documents\Face recognition"
model_path = f"{base_path}\\Result_Training_Hourglass\\final_model.pth"
work_dataset_path = f"{base_path}\\Work_Dataset"

model = StackedHourglassNetwork(num_stacks=2, channels=128, input_size=112)

checkpoint = torch.load(model_path, map_location='cpu')

if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import cv2
import shutil

def get_keypoints_from_heatmaps(heatmaps):
    batch_size, num_keypoints, H, W = heatmaps.shape
    coords = torch.zeros(batch_size, num_keypoints, 2)
    for b in range(batch_size):
        for k in range(num_keypoints):
            idx = torch.argmax(heatmaps[b, k])
            y = idx // W
            x = idx % W
            coords[b, k, 0] = x
            coords[b, k, 1] = y
    return coords.float()

def align_face_numpy(image_rgb, keypoints, output_size=(112, 112)):
    if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
        pass
    elif len(image_rgb.shape) == 3 and image_rgb.shape[0] == 3:
        image_rgb = np.transpose(image_rgb, (1, 2, 0))
    
    image_bgr = image_rgb[:, :, ::-1] 
    
    # Определяем левый/правый глаз по X
    eye1, eye2 = keypoints[0], keypoints[1]
    if eye1[0] > eye2[0]:
        left_eye, right_eye = eye2, eye1
    else:
        left_eye, right_eye = eye1, eye2
    
    # Вычисляем угол поворота
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Центр между глазами
    center = (left_eye + right_eye) * 0.5
    
    eye_distance = np.linalg.norm(right_eye - left_eye)
    desired_distance = 0.45 * output_size[0]
    scale = desired_distance / eye_distance
    
    # Матрица поворота
    M = cv2.getRotationMatrix2D(tuple(center), angle, scale)
    
    # Центрируем по вертикали
    target_y = output_size[1] * 0.35
    eyes_center_y = center[1]
    M[1, 2] += target_y - eyes_center_y * scale
    
    aligned_bgr = cv2.warpAffine(image_bgr, M, output_size, 
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REFLECT_101)
    
    aligned_rgb = aligned_bgr[:, :, ::-1]  
    
    return aligned_rgb

def predict_keypoints_and_align_all(input_dir, model, output_dir, device, batch_size=8, output_size=(112, 112)):
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    image_files.sort()
    
    
    processed = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i+batch_size]
            
            batch_images = []
            for img_file in batch_files:
                img_path = os.path.join(input_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize(output_size, Image.BICUBIC)
                img_array = np.array(img_resized).transpose(2, 0, 1) / 255.0
                batch_images.append(img_array)
                img.close()
            
            if not batch_images:
                continue
                
            batch_tensor = torch.FloatTensor(np.array(batch_images))
            batch_tensor = batch_tensor.to(device)
            
            model.eval()
            predictions = model(batch_tensor)
            pred_heatmaps = predictions[-1].cpu()
            pred_coords = get_keypoints_from_heatmaps(pred_heatmaps)
            
            for j, img_file in enumerate(batch_files):
                orig_rgb_hw = (batch_images[j] * 255.0).astype(np.uint8).transpose(1, 2, 0)
                keypoints = pred_coords[j].numpy()
                
                aligned_rgb = align_face_numpy(orig_rgb_hw, keypoints, output_size)
                
                output_path = os.path.join(output_dir, img_file)
                Image.fromarray(aligned_rgb).save(output_path)
                processed += 1
    
    return output_dir

if __name__ == "__main__":
    base_path = r"C:\Users\Public\Documents\Face recognition"
    input_dir = f"{base_path}\\Work_Dataset"
    output_dir = f"{base_path}\\Final_Aligned_Dataset"
    
    device = torch.device('cpu')
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    predict_keypoints_and_align_all(input_dir, model, output_dir, device, batch_size=8)
