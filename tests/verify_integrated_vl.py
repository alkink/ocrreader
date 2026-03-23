from pathlib import Path
import cv2

from ocrreader.config import OCRConfig
from ocrreader.ocr_engine import create_glm_fallback_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = PROJECT_ROOT / "testdata" / "WhatsApp_Image_2026-03-03_at_18.31.01.jpeg"

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
    
    if not SAMPLE_IMAGE.exists():
        print(f"Image not found: {SAMPLE_IMAGE}")
        return
        
    img = cv2.imread(str(SAMPLE_IMAGE))
    
    print(f"Running inference on real image: {SAMPLE_IMAGE}")
    res = engine.read_text(img)
    print(f"\n--- Result ---\n{res}\n--- End ---")

if __name__ == '__main__':
    main()
