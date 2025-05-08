import os
from rapidocr import RapidOCR

def process_image_with_ocr(input_path, output_path=None):
    engine = RapidOCR()
    result = engine(input_path)

    if output_path is None:
        base_name = os.path.basename(input_path)
        file_name, file_ext = os.path.splitext(base_name)
        output_path = os.path.join("media", f"{file_name}_rapidocr{file_ext}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result.vis(output_path)

    return result

# Example usage
if __name__ == "__main__":
    img_path = os.path.join("media", "quotes.png")
    out_path = os.path.join("media", "quotes_result.png")

    result = process_image_with_ocr(img_path, out_path)

    print(result)