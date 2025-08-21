from ultralytics import YOLO

# Load the YOLO model (replace 'yolov8n.pt' with your model file if different)
model = YOLO("yolov8n.pt")

# Run prediction on the image
detection_output = model.predict(
    source=r"C:\Users\anish\Data Science with Gen AI\YOLO\img\two.jpg",
    conf=0.25,  # Confidence threshold
    save=True   # Save output (e.g., annotated image)
)

# Optional: Print or process the results
print(detection_output)