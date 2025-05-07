from ultralytics import YOLO
import cv2
import os
import numpy as np

def detect_and_count(image_input, output_path=None):

    model_path = os.path.join('models', 'yolo11n.onnx')
    model_onnx = YOLO(model_path)

    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError("image_input must be either a file path (str) or an image array (numpy.ndarray)")

    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.join('media', 'out.png')

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run inference
    results = model_onnx(img)

    # Draw bounding boxes and labels
    for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                              results[0].boxes.cls.cpu().numpy(),
                              results[0].boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model_onnx.names[int(cls)]} {conf:.2f}" if hasattr(model_onnx, 'names') else str(int(cls))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save output image
    cv2.imwrite(output_path, img)

    # Create summary
    classes = results[0].boxes.cls.cpu().numpy()
    names = model_onnx.names if hasattr(model_onnx, 'names') else {}

    count_dict = {}
    for cls in classes:
        cls = int(cls)
        count_dict[cls] = count_dict.get(cls, 0) + 1

    summary_parts = []
    for cls_id in count_dict:
        label = names.get(cls_id, f'class_{cls_id}')
        count = count_dict[cls_id]
        summary_parts.append(f"{count} {label}")

    summary_string = ", ".join(summary_parts)

    return summary_string, img

if __name__ == "__main__":
    image_path = os.path.join('media', 'dogs.png')
    output_path = os.path.join('media', 'dogs_out.png')
    summary, _ = detect_and_count(image_path, output_path=output_path)
    print(f"Summary: {summary}")


