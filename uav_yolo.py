
from ultralytics import YOLO
import cv2
import time
import torch

def main():
    # Check CUDA
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    # Load model
    model = YOLO('yolov8n.pt')
    model.to(device)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cam ok")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inference
        results = model.predict(source=frame, conf=0.5, iou=0.45, device=device, verbose=False)
        annotated_frame = results[0].plot()
        
        # Calculate FPS
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        
        # Display on frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Objects: {len(results[0].boxes)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Print detections to terminal
        if len(results[0].boxes) > 0:
            detections = []
            for box in results[0].boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                detections.append(f"{cls_name}({conf:.2f})")
            print(f"FPS: {fps:.1f} | Detected: {', '.join(detections)}")
        
        # Show
        cv2.imshow("UAV YOLO-Nano", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
