from flask import Flask, render_template, request, Response, jsonify
import cv2
import base64
from ultralytics import YOLO
import numpy as np

app = Flask(__name__ ,static_url_path='/static')
model = YOLO('best.pt')
print('Loaded model class names:', model.names)

# Author: tithi bose

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Run YOLO detection with lower confidence threshold
        results = model(frame, conf=0.3)
        # Draw results on frame
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    result = model.predict(source=0, imgsz=640, conf=0.6, show=True)
    return result.imgs[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    # Remove header (data:image/jpeg;base64,...)
    img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    try:
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = model(img, conf=0.3)
        predictions = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                predictions.append({
                    'label': label,
                    'confidence': conf,
                    'box': [x1, y1, x2, y2]
                })
        return jsonify({'success': True, 'prediction': predictions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

