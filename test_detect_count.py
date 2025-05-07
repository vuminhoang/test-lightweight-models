from ultralytics import YOLO
import cv2
import os

# Set relative path
model_path = os.path.join('models', 'yolo11n.onnx')
model_onnx = YOLO(model_path)

# Read input image
img = cv2.imread(os.path.join('media', 'dogs.png'))
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
cv2.imwrite(os.path.join('media', 'dogs_out.png'), img)

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

print(f"Summary: {summary_string}")
