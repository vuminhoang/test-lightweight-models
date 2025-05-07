from ultralytics import YOLO
import cv2
import os
import numpy as np

model_path = os.path.join('models', 'yolo11n.onnx')
model_onnx = YOLO(model_path)

def resize_image(img, max_size=800):
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))

def detect_and_count(image_input, model_onnx, output_path=None, resize=True):
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError("image_input must be either a file path (str) or an image array (numpy.ndarray)")

    if resize:
        img = resize_image(img)

    results = model_onnx(img)
    names = model_onnx.names if hasattr(model_onnx, 'names') else {}

    if hasattr(results[0].boxes, 'xyxy'):
        for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                                  results[0].boxes.cls.cpu().numpy(),
                                  results[0].boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names.get(int(cls), f'class_{int(cls)}')} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_path, img)

    # Summary
    count_dict = {}
    if hasattr(results[0].boxes, 'cls'):
        classes = results[0].boxes.cls.cpu().numpy()
        for cls in classes:
            cls = int(cls)
            count_dict[cls] = count_dict.get(cls, 0) + 1

    summary_parts = [f"{count} {names.get(cls_id, f'class_{cls_id}')}" for cls_id, count in count_dict.items()]
    summary_string = ", ".join(summary_parts)

    return summary_string, img

if __name__ == "__main__":
    image_path = os.path.join('media', 'dogs.png')
    output_path = os.path.join('media', 'dogs_out.png')
    summary, _ = detect_and_count(image_path, model_onnx, output_path=output_path)
    print(f"Summary: {summary}")
