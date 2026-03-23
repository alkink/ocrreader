from ocrreader.config import OCRConfig
from ocrreader.ocr_engine import create_glm_fallback_engine
import numpy as np
import cv2
import os
import json

def main():
    config = OCRConfig(
        engine="paddle",
        glm_fallback_enabled=True,
        glm_model="paddle_vl"
    )
    
    # We want to see default output
    engine = create_glm_fallback_engine(config)
    
    img_path = r'dataset\photo\132020380_1753782264797081_6362873494041143451_n.jpg'
    img = cv2.imread(img_path)
    
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
        
    with open('tests/vl_inspection_result.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("Done. Results saved to tests/vl_inspection_result.json")

if __name__ == '__main__':
    main()
