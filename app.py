from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

# =========================
# CONFIG
# =========================
ONNX_MODEL_PATH = "models/models.onnx"

CLASS_NAMES = [
    'ButterMilk',
    'Chai_mazza',
    'Gold',
    'Shakti',
    'SnT',
    'TSpecial',
    'Tadka_Chaas',
    'Taza'
]

IMAGE_SIZE = 224


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="ONNX Image Classification API",
    version="1.0.0"
)


# =========================
# LOAD ONNX MODEL
# =========================
session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)


# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Must exactly match training preprocessing:
    - Resize to 224x224
    - Convert to RGB
    - Normalize to [0, 1]
    - CHW format
    """
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    img = np.array(image).astype(np.float32)
    img = img / 255.0                     # normalize
    img = np.transpose(img, (2, 0, 1))    # HWC → CHW
    img = np.expand_dims(img, axis=0)     # add batch dim

    return img

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    input_tensor = preprocess_image(image)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run(
        [output_name],
        {input_name: input_tensor}
    )

    logits = outputs[0][0]  # shape: (8,)

    # Softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    return {
        "filename": file.filename,
        "predicted_class": CLASS_NAMES[class_id],
        "confidence": round(confidence, 4),
        "class_id": class_id,
        "all_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }
    }


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "ONNX inference API running"}
