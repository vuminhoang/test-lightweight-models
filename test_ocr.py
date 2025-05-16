import os
from rapidocr import RapidOCR

def get_ocr_engine():
    config_path = os.path.join("configs", "config_rapid_ocr.yml")
    return RapidOCR(config_path)

def process_image_with_ocr(engine, input_path, output_path=None):
    result = engine(input_path)

    if output_path is None:
        base_name = os.path.basename(input_path)
        file_name, file_ext = os.path.splitext(base_name)
        output_path = os.path.join("media", f"{file_name}_rapidocr{file_ext}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.vis(output_path)

    return result

if __name__ == "__main__":
    img_path = os.path.join("media", "test_ocr_eng.jpg")
    out_path = os.path.join("output", "test_ocr_rapidocr.jpg")

    engine = get_ocr_engine()
    result = process_image_with_ocr(engine, img_path, out_path)

    print(result)
    print("----------------")
    print(result.txts)
