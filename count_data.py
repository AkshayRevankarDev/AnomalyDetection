import os
import glob

def get_paths(dataset_type, raw_path):
    if dataset_type == "LUNG":
        normal_pattern = ["NORMAL"]
        sick_pattern = ["PNEUMONIA"]
    elif dataset_type == "BRAIN":
        # Combined dataset: Standard 'no/yes' AND 'brain_mri_4class'
        normal_pattern = ["no", "NO", "notumor"]
        sick_pattern = ["yes", "YES", "glioma", "meningioma", "pituitary"]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
    normal_images = []
    for pattern in normal_pattern:
        normal_images.extend(glob.glob(os.path.join(raw_path, "**", pattern, "*.jpeg"), recursive=True))
        normal_images.extend(glob.glob(os.path.join(raw_path, "**", pattern, "*.jpg"), recursive=True))
        normal_images.extend(glob.glob(os.path.join(raw_path, "**", pattern, "*.png"), recursive=True))
        
    sick_images = []
    for pattern in sick_pattern:
        sick_images.extend(glob.glob(os.path.join(raw_path, "**", pattern, "*.jpeg"), recursive=True))
        sick_images.extend(glob.glob(os.path.join(raw_path, "**", pattern, "*.jpg"), recursive=True))
        sick_images.extend(glob.glob(os.path.join(raw_path, "**", pattern, "*.png"), recursive=True))
        
    return list(set(normal_images)), list(set(sick_images))

normal, sick = get_paths("BRAIN", "data/raw")
print(f"Normal: {len(normal)}")
print(f"Sick: {len(sick)}")
print(f"Total: {len(normal) + len(sick)}")
