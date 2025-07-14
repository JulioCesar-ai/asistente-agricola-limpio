from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
from PIL import Image
import io
import base64
from ultralytics import YOLO
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# ✅ Configurar logging
# ----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------------
# ✅ Cargar modelo fine-tuneado desde Hugging Face
# ----------------------------
try:
    logger.info("Cargando modelo Julio1023/phi2-merged...")
    tokenizer = AutoTokenizer.from_pretrained("Julio1023/phi2-merged")
    model = AutoModelForCausalLM.from_pretrained("Julio1023/phi2-merged", torch_dtype=torch.float16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("✅ Modelo cargado correctamente.")
except Exception as e:
    logger.error(f"❌ Error al cargar el modelo: {e}")
    raise

# ----------------------------
# ✅ Función de respuesta tipo Alpaca
# ----------------------------
def get_answer(question):
    try:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Responde la siguiente pregunta de agricultura.

### Input:
{question}

### Response:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Respuesta generada: {respuesta}")
        return respuesta.split("### Response:")[-1].strip()
    except Exception as e:
        logger.error(f"❌ Error generando respuesta: {e}")
        return "[sin respuesta]"

# ----------------------------
# ✅ Cargar modelos YOLO
# ----------------------------
try:
    logger.info("Cargando modelos YOLO...")
    models = {
        "yolov5s": YOLO("yolov5s.pt"),
        "yolov5m": YOLO("yolov5m.pt")
    }
    logger.info("✅ Modelos YOLO cargados.")
except Exception as e:
    logger.error(f"❌ Error cargando modelos YOLO: {e}")
    models = {}

# ----------------------------
# ✅ Inicializar app Flask
# ----------------------------
app = Flask(__name__, static_folder='public')
CORS(app)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        pregunta = data.get('pregunta')
        if not pregunta:
            return jsonify({"respuesta": "Error: No se proporcionó una pregunta."}), 400

        respuesta = get_answer(pregunta)
        if respuesta == "[sin respuesta]":
            respuesta = "Lo siento, no pude encontrar una respuesta adecuada."

        return jsonify({"respuesta": respuesta})
    except Exception as e:
        logger.error(f"❌ Error en /chat: {e}")
        return jsonify({"respuesta": "Error al obtener respuesta del modelo."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form.get('model')
        file = request.files.get('image')

        if not model_name or model_name not in models:
            return jsonify({"error": "Modelo YOLO no válido."}), 400
        if not file:
            return jsonify({"error": "Imagen no proporcionada."}), 400

        image = Image.open(io.BytesIO(file.read()))
        results = models[model_name](image)
        annotated_img = results[0].plot()

        buffered = io.BytesIO()
        Image.fromarray(annotated_img).save(buffered, format="JPEG")
        encoded_img = base64.b64encode(buffered.getvalue()).decode()
        detections = json.loads(results[0].tojson()) if results[0].boxes else []

        return jsonify({
            "image": encoded_img,
            "detections": detections
        })
    except Exception as e:
        logger.error(f"❌ Error en /predict: {e}")
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500

# ----------------------------
# ✅ Ejecutar servidor
# ----------------------------
if __name__ == '__main__':
    app.run(port=5000, debug=True)
