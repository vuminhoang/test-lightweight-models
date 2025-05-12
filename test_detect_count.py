import os
import numpy as np
import torch
import torchvision
import cv2
from ultralytics import YOLO

model_path = os.path.join('models', 'yolo11n.onnx')
model_onnx = YOLO(model_path)

COLOR_LIST = [
    (0, 255, 0),    # Green
    (0, 255, 255),  # Yellow
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 0, 255),  # Magenta
    (255, 165, 0),  # Orange
    (0, 128, 255),  # Light Blue
    (128, 0, 255),  # Purple
    (0, 255, 128),  # Spring Green
    (255, 192, 203) # Pink
]
COLOR_PALETTE = {}

def get_color_for_class(cls):
    if cls not in COLOR_PALETTE:
        COLOR_PALETTE[cls] = COLOR_LIST[cls % len(COLOR_LIST)]
    return COLOR_PALETTE[cls]

def resize_image(img, max_size=800):
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))

def apply_nms(boxes, scores, iou_threshold=0.5):
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    return keep_indices.numpy()

def detect_and_count(image_input, model_onnx, output_path=None, resize=True):
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()

    if resize:
        img = resize_image(img)

    results = model_onnx(img)
    names = model_onnx.names if hasattr(model_onnx, 'names') else {}

    count_dict = {}

    if hasattr(results[0].boxes, 'xyxy'):
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        keep = apply_nms(boxes, scores, iou_threshold=0.5)

        for idx in keep:
            x1, y1, x2, y2 = map(int, boxes[idx])
            cls = int(classes[idx])
            score = scores[idx]

            label = f"{names.get(cls, f'class_{cls}')} {score:.2f}"
            color = get_color_for_class(cls)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            count_dict[cls] = count_dict.get(cls, 0) + 1

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_path, img)

    summary_parts = [f"{count} {names.get(cls_id, f'class_{cls_id}')}" for cls_id, count in count_dict.items()]
    summary_string = ", ".join(summary_parts)

    return summary_string, img

if __name__ == "__main__":
    image_path = os.path.join('media', 'test_8.jpg')
    output_path = os.path.join('output', 'test_8_out.png')
    summary, _ = detect_and_count(image_path, model_onnx, output_path=output_path)
    print(f"Summary: {summary}")