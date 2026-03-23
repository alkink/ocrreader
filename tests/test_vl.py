import os
import glob

# Explicitly add all known nvidia/cuda DLL directories
_nvidia_bin_dirs = [
    r"C:\Users\alkin\miniconda3\Lib\site-packages\nvidia\cu13\bin",
    r"C:\Users\alkin\miniconda3\Lib\site-packages\nvidia\cudnn\bin",
    r"C:\Users\alkin\AppData\Roaming\Python\Python313\site-packages\nvidia\cublas\bin",
    r"C:\Users\alkin\AppData\Roaming\Python\Python313\site-packages\nvidia\cuda_runtime\bin",
    r"C:\Users\alkin\AppData\Roaming\Python\Python313\site-packages\nvidia\cudnn\bin",
]
for _p in _nvidia_bin_dirs:
    if os.path.isdir(_p):
        os.add_dll_directory(_p)
        print(f"[DLL] Added: {_p}")
    else:
        print(f"[DLL] NOT found: {_p}")

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCRVL
import numpy as np

def main():
    print("Loading PaddleOCRVL model...")
    vl = PaddleOCRVL(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_ocr_for_image_block=False,
    )
    print("Model loaded OK!")
    img = np.ones((100, 200, 3), dtype=np.uint8) * 200
    res = vl.predict(img, prompt_label="text")
    print("Result:", res)

if __name__ == '__main__':
    main()
