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
    
    print("Initializing integrated PaddleOCRVLEngine (Layout ENABLED)...")
    # We'll temporarily modify the engine to enable layout for this test
    engine = create_glm_fallback_engine(config)
    engine._vl = engine._vl.__class__(
        use_layout_detection=True,
        use_ocr_for_image_block=True
    )
    
    img = cv2.imread(str(SAMPLE_IMAGE))
    
    print("Running inference...")
    res = engine.read_text(img)
    print(f"\n--- Result ---\n{res}\n--- End ---")

if __name__ == '__main__':
    main()
