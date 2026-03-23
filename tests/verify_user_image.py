from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocrreader.config import OCRConfig
from ocrreader.ocr_engine import create_glm_fallback_engine
import numpy as np
import cv2
import os

def main():
    config = OCRConfig(
        engine="paddle",
        glm_fallback_enabled=True,
        glm_model="paddle_vl"
    )
    
    print("Initializing integrated PaddleOCRVLEngine...")
    engine = create_glm_fallback_engine(config)
    if engine is None:
        print("Failed to create engine")
        return
        
    img_path = r'c:\Users\alkin\projects\ocrreader\testdata\WhatsApp_Image_2026-03-03_at_18.31.01.jpeg'
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
        
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
        
    print(f"Running inference on image: {img_path}")
    res = engine.read_text(img)
    
    # Save the result to a UTF-8 encoded file
    with open('tests/user_image_full_res.txt', 'w', encoding='utf-8') as f:
        f.write(res)
    
    print("Done. Result saved to tests/user_image_full_res.txt")

if __name__ == '__main__':
    main()
