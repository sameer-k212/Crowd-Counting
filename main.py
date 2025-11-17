from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = FastAPI()

# Load YOLO model
model = YOLO('yolov8n.pt')

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run YOLO detection
        results = model(img)
        
        # Count persons (class 0 in COCO dataset)
        person_count = 0
        annotated_img = img.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        person_count += 1
                        # Draw rectangle with bold green border
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 6)
                        
                        # Add text background for better visibility
                        label = f'Person {person_count}'
                        font_scale = 1.2
                        thickness = 3
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        cv2.rectangle(annotated_img, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1)
                        cv2.putText(annotated_img, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "person_count": person_count,
            "output_image": f"data:image/jpeg;base64,{img_base64}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)