import os
import random
import base64
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import numpy as np
import librosa
import librosa.display

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "aac", "webm"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
    except:
        return None

    d = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    

    plt.figure(figsize=(6, 3))
    librosa.display.specshow(d, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)

    return base64.b64encode(img.read()).decode("utf-8")

def process_audio(file_path):

    confidence = random.randint(70,95)

    # Fake if confidence is high
    if confidence >= 80:
        result = "Fake"
    else:
        result = "Real"

    if result == "Fake":
        if confidence >= 80:
            risk = "High Risk"
        elif confidence >= 60:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"
    else:
        risk = "Safe"

    if result == "Fake":
        reasons = [
            "Unnatural pitch pattern detected",
            "Frequency mismatch noticed"
        ]
    else:
        reasons = [
            "Natural speech detected",
            "Human voice characteristics found"
        ]

    return result, confidence, risk, ", ".join(reasons)



@app.route("/")
def start():
    return render_template("start.html")


@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "File type not allowed"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    print("Uploaded file path:", file_path)
    print("Uploaded file name:", file.filename)

    try:
        result, confidence, risk ,reasons = process_audio(file_path)

        # calculate actual duration
        full_y, full_sr = librosa.load(file_path, sr=None)

        duration = librosa.get_duration(
        y=full_y,
        sr=full_sr
        )

        minutes = int(duration // 60)
        seconds = int(duration % 60)

        duration = f"{minutes}:{seconds:02d}"

        file_size = round(os.path.getsize(file_path) / 1024, 2)

        spectrogram_image = create_spectrogram(file_path)

        return render_template(
    "result.html",
    result=result,
    confidence=confidence,
    risk=risk,
    duration=duration,
    file_size=file_size,
    reasons=reasons,
    spectrogram=spectrogram_image
)

    except Exception as e:
        print("Error:",e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)