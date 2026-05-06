from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import cv2

app = Flask(__name__)
CORS(app)

# 1. Load the Model
try:
    model = load_model("recognition.keras")
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error: Could not load recognition.keras. {e}")

def preprocess(image):
    image = image.convert("L")
    img = np.array(image)
    img = 255 - img 
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    digits = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            digit = thresh[y:y+h, x:x+w]
            digit = cv2.copyMakeBorder(digit, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
            digit = cv2.resize(digit, (28, 28))
            digit = digit.astype('float32') / 255.0
            digit = digit.reshape(1, 28, 28, 1)
            digits.append(digit)
    return digits

@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>AI Digit Recognizer</title>
        <link href="https://googleapis.com" rel="stylesheet">
        <style>
            body { margin:0; font-family: 'Poppins', sans-serif; background: #121212; color:white; text-align:center; min-height:100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; }
            h1 { font-size:32px; margin-bottom: 20px; color: #00c9a7; }
            canvas { background:white; border-radius:15px; box-shadow:0 10px 30px rgba(0,0,0,0.5); cursor:crosshair; touch-action: none; }
            .buttons { margin-top: 20px; }
            button { margin:10px; padding:12px 30px; border:none; border-radius:8px; font-size:16px; font-weight:bold; cursor:pointer; transition: 0.2s; }
            .predict { background:#00c9a7; color:white; }
            .predict:active { transform: scale(0.95); }
            .clear { background:#ff6b6b; color:white; }
            .result-box { margin-top:30px; font-size:18px; padding:20px 30px; background:rgba(255,255,255,0.1); border-radius:12px; border: 1px solid rgba(255,255,255,0.2); min-width: 300px; line-height: 1.6; }
            b { color: #00c9a7; font-size: 24px; }
        </style>
    </head>
    <body>
        <h1>🤖 AI Digit Recognizer</h1>
        <canvas id="canvas" width="500" height="200"></canvas>
        <div class="buttons">
            <button class="predict" onclick="predict()">🔮 Predict</button>
            <button class="clear" onclick="clearCanvas()">🗑️ Clear</button>
        </div>
        <div class="result-box" id="result">Draw digits clearly!</div>

        <script>
            let canvas = document.getElementById("canvas");
            let ctx = canvas.getContext("2d");
            let drawing = false;

            function clearCanvas(){
                ctx.fillStyle="white";
                ctx.fillRect(0,0,canvas.width,canvas.height);
                document.getElementById("result").innerHTML = "Draw digits clearly!";
            }
            clearCanvas();

            canvas.onmousedown = (e) => { drawing = true; draw(e); };
            canvas.onmouseup = () => { drawing = false; ctx.beginPath(); };
            canvas.onmousemove = (e) => { if(drawing) draw(e); };

            // Support for touch devices
            canvas.ontouchstart = (e) => { e.preventDefault(); drawing = true; draw(e.touches[0]); };
            canvas.ontouchend = () => { drawing = false; ctx.beginPath(); };
            canvas.ontouchmove = (e) => { e.preventDefault(); if(drawing) draw(e.touches[0]); };

            function draw(e) {
                let rect = canvas.getBoundingClientRect();
                ctx.lineWidth = 14;
                ctx.lineCap = "round";
                ctx.strokeStyle = "black";
                ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
            }

            function predict(){
                let image = canvas.toDataURL("image/png");
                document.getElementById("result").innerHTML = "⌛ Predicting...";

                fetch("/predict", {
                    method:"POST",
                    headers:{"Content-Type":"application/json"},
                    body:JSON.stringify({image:image})
                })
                .then(res => res.json())
                .then(data => {
                    if(data.error){
                        document.getElementById("result").innerText = "Error: " + data.error;
                        return;
                    }
                    if(!data.number || data.number === ""){
                        document.getElementById("result").innerText = "No digits found";
                        return;
                    }

                    // 🎯 Format confidence nicely
                    let confidenceText = "";
                    if(data.confidence){
                        confidenceText = data.confidence.map((c, i) => 
                            "Digit " + data.number[i] + ": " + c + "%"
                        ).join("<br>");
                    }

                    document.getElementById("result").innerHTML = `
                        <b>Result: ${data.number}</b><br><br>
                        <strong>Confidence:</strong><br>${confidenceText}
                    `;
                });
            }
        </script>
    </body>
    </html>
    """

@app.route("/predict", methods=["POST"])
def predict_digits():
    try:
        data = request.json
        img_str = data["image"].split(",")[1]
        img_data = base64.b64decode(img_str)
        image = Image.open(BytesIO(img_data))
        
        digits = preprocess(image)
        if not digits:
            return jsonify({"number": "", "confidence": []})

        result = ""
        confidences = []

        for d in digits:
            pred = model.predict(d, verbose=0)[0]
            digit = int(np.argmax(pred))
            confidence = float(np.max(pred)) * 100
            result += str(digit)
            confidences.append(round(confidence, 2))

        return jsonify({"number": result, "confidence": confidences})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
