import os
from PIL import Image
import numpy as np

def create_dummy_data():
    base_path = "data/raw/brain_mri"
    categories = ["yes", "no"]
    
    for cat in categories:
        path = os.path.join(base_path, cat)
        os.makedirs(path, exist_ok=True)
        
        print(f"Creating 20 dummy images in {path}...")
        for i in range(20):
            # Create random noise image
            img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(path, f"dummy_{i}.jpg"))

if __name__ == "__main__":
    create_dummy_data()
