import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

# 🔧 --- PATCH: remove 'quantization_config' during deserialization ---
from tensorflow.keras.layers import Dense as _Dense
_old_from_config = _Dense.from_config

@classmethod
def _patched_from_config(cls, config):
    if isinstance(config, dict) and 'quantization_config' in config:
        config.pop('quantization_config', None)
    return _old_from_config.__func__(cls, config)

_Dense.from_config = _patched_from_config
# 🔧 --- END PATCH ---

app = Flask(__name__)

# Load model (your file name)
try:
    model = tf.keras.models.load_model("cnn_clean.h5", compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None

live = {
    "temperature": 30,
    "voltage": 48,
    "current": 10,
    "battery": 90,
    "status": "SAFE"
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/update', methods=['POST'])
def update():
    try:
        temp = float(request.form['temperature'])
        volt = float(request.form['voltage'])
        curr = float(request.form['current'])

        if model:
            x = np.zeros((1, 100, 5))
            x[0, 0, 0] = temp
            x[0, 0, 1] = volt
            x[0, 0, 2] = curr
            pred = model.predict(x, verbose=0)
            score = float(pred[0][0])
        else:
            score = 0.0

        if temp >= 50 or score > 0.7:
            status = "DANGER"
        elif temp >= 40 or score > 0.4:
            status = "MEDIUM"
        else:
            status = "SAFE"

        battery = max(0, 100 - temp)

        live.update({
            "temperature": round(temp, 2),
            "voltage": round(volt, 2),
            "current": round(curr, 2),
            "battery": round(battery, 2),
            "status": status
        })
        return "OK"

    except Exception as e:
        print("Update error:", e)
        return "ERROR"

@app.route('/data')
def data():
    return jsonify(live)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)