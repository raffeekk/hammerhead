# model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# model.train(data=r'C:\Users\l1kr1\Downloads\adult_filter',
#             epochs=1, imgsz=64)

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r'C:\Users\l1kr1\runs\classify\train2\weights\last.pt')

# Run inference on 'bus.jpg' with arguments
model.predict(r'C:\Users\l1kr1\OneDrive\Documents\hahahahahahahahaton\images.jpg', save=True, imgsz=640, conf=0.5)
