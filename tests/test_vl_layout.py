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
    
    print("Initializing integrated PaddleOCRVLEngine (Layout ENABLED)...")
    # We'll temporarily modify the engine to enable layout for this test
    engine = create_glm_fallback_engine(config)
    engine._vl = engine._vl.__class__(
        use_layout_detection=True,
        use_ocr_for_image_block=True
    )
    
    img_path = r'dataset/generated/qa/benchmark/benchmark_images/132020380_1753782264797081_6362873494041143451_n.jpg'
    img = cv2.imread(img_path)
    
    print("Running inference...")
    res = engine.read_text(img)
    print(f"\n--- Result ---\n{res}\n--- End ---")

if __name__ == '__main__':
    main()
