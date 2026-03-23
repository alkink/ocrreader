from pathlib import Path
import json
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
    
    # We want to see default output
    engine = create_glm_fallback_engine(config)
    
    img = cv2.imread(str(SAMPLE_IMAGE))
    
    print("Running VL predict...")
    res = engine._vl.predict(img)
    
    output_data = []
    for i, r in enumerate(res):
        item = {"index": i, "type": str(type(r))}
        if hasattr(r, 'to_dict'):
            d = r.to_dict()
            item["data"] = d
        else:
            item["str_val"] = str(r)
        output_data.append(item)
        
    with open(PROJECT_ROOT / 'tests' / 'vl_inspection_result.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("Done. Results saved to tests/vl_inspection_result.json")

if __name__ == '__main__':
    main()
