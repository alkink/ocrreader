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
        
    print(f"Engine created: {type(engine).__name__}")
    
    # Use a valid image path found on disk
    img_path = r'dataset\photo\132020380_1753782264797081_6362873494041143451_n.jpg'
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
        
    img = cv2.imread(img_path)
    
    print(f"Running inference on real image: {img_path}")
    res = engine.read_text(img)
    print(f"\n--- Result ---\n{res}\n--- End ---")

if __name__ == '__main__':
    main()
