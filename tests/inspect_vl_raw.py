from pathlib import Path
import json
import cv2

from ocrreader.config import OCRConfig
from ocrreader.ocr_engine import create_glm_fallback_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = PROJECT_ROOT / "testdata" / "WhatsApp_Image_2026-03-03_at_18.31.01.jpeg"

def json_serializable(obj):
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return str(obj)

def main():
    config = OCRConfig(
        engine="paddle",
        glm_fallback_enabled=True,
        glm_model="paddle_vl"
    )
    
    print("Initializing integrated PaddleOCRVLEngine...")
    engine = create_glm_fallback_engine(config)
    
    img = cv2.imread(str(SAMPLE_IMAGE))
    
    # Access the private _vl to call predict directly and see raw output
    print("Running raw VL predict...")
    raw_res = engine._vl.predict(img)
    
    # Save the raw result structure for inspection
    with open(PROJECT_ROOT / 'tests' / 'vl_raw_res.json', 'w', encoding='utf-8') as f:
        # Convert paddlex result objects to dicts if possible
        serializable_res = []
        for r in raw_res:
            if hasattr(r, 'to_dict'):
                serializable_res.append(r.to_dict())
            else:
                serializable_res.append(str(r))
        json.dump(serializable_res, f, indent=2, ensure_ascii=False)
    
    print("Raw result saved to tests/vl_raw_res.json")

if __name__ == '__main__':
    main()
