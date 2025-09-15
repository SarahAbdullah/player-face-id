import os
import io
import numpy as np
import streamlit as st
import cv2
import torch
from joblib import load
from facenet_pytorch import MTCNN, InceptionResnetV1

st.set_page_config(page_title="Player Face ID ", page_icon="⚽", layout="centered")
st.title("⚽ Player Face Identification")
st.caption("OpenCV for I/O & drawing • MTCNN for detection • FaceNet embeddings • the trained classifier (kNN)")

# ----------------------------
# Paths
# ----------------------------
MODELS_DIR = "models"
CLF_PATH   = os.path.join(MODELS_DIR, "player_id.joblib")
LBL_PATH   = os.path.join(MODELS_DIR, "labels.joblib")
EMB_PATH   = os.path.join(MODELS_DIR, "embeddings.npz")  #for cosine-sim "Unknown"

# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_face_models(device_str="cpu"):
    device = torch.device(device_str)
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    # warm-up to avoid slow first call
    with torch.inference_mode():
        _ = embedder(torch.zeros(1, 3, 160, 160, device=device))
    return mtcnn, embedder, device

@st.cache_resource(show_spinner=False)
def load_classifier():
    if not (os.path.exists(CLF_PATH) and os.path.exists(LBL_PATH)):
        raise FileNotFoundError(
            "Missing trained classifier files. "
            "Add models/player_id.joblib and models/labels.joblib to the repo."
        )
    clf = load(CLF_PATH)
    le  = load(LBL_PATH)
    return clf, le

@st.cache_resource(show_spinner=False)
def load_centroids():
    """
    If models/embeddings.npz exists, compute class centroids (L2-normalized)
    for cosine-similarity open-set checks. Returns (C, names) or (None, None).
      C: (K, 512) array of centroids
      names: list[str] class names in same order as rows of C
    """
    if not os.path.exists(EMB_PATH):
        return None, None
    data = np.load(EMB_PATH, allow_pickle=True)
    X, y = data["X"], data["y"].astype(str)
    classes = sorted(set(y.tolist()))
    centroids, names = [], []
    for cls in classes:
        m = X[y == cls].mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)  # normalize for cosine sim
        centroids.append(m)
        names.append(cls)
    C = np.vstack(centroids)
    return C, names

mtcnn, embedder, device = load_face_models("cpu")  # Streamlit Cloud runs on CPU
clf, le = load_classifier()
C, centroid_names = load_centroids()  # optional

# ----------------------------
# Utilities (OpenCV I/O)
# ----------------------------
def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    """Convert Streamlit UploadedFile to OpenCV BGR image."""
    data = uploaded_file.read()
    img_array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to decode image")
    return bgr

def draw_label(bgr: np.ndarray, x1, y1, text: str, color=(0, 255, 0)):
    """Draw a filled label box + text (OpenCV)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(bgr, (x1, y0), (x1 + tw + 8, y1), color, -1)    # filled bg
    cv2.putText(bgr, text, (x1 + 4, y1 - 6), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

@torch.inference_mode()
def predict_bgr(
    bgr: np.ndarray,
    prob_threshold: float = 0.60,
    use_distance_check: bool = False,
    sim_threshold: float = 0.45
):
    """
    Predict on a BGR image and return (annotated_bgr, results).
    Unknown if:
      - top class probability < prob_threshold  OR
      - (optional) max cosine similarity to any class centroid < sim_threshold
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 1) Detect boxes (for drawing)
    boxes, _ = mtcnn.detect(rgb)

    # 2) Get aligned faces for embeddings
    faces, _ = mtcnn(rgb, return_prob=True)
    results = []
    if faces is None or (isinstance(faces, torch.Tensor) and faces.ndim == 0):
        return bgr, results
    if isinstance(faces, torch.Tensor) and faces.ndim == 3:
        faces = faces.unsqueeze(0)  # single face -> batch

    # 3) Embeddings
    embs = embedder(faces.to(device)).cpu().numpy()  # (N, 512)

    # 4) Classifier probabilities
    proba = clf.predict_proba(embs)                  # (N, C)
    idx   = np.argmax(proba, axis=1)
    names = le.inverse_transform(idx)
    confs = proba[np.arange(len(idx)), idx]

    # 5) Optional cosine similarity check to centroids
    max_sims = None
    if use_distance_check and C is not None:
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        sims = embs_norm @ C.T                       # (N, K)
        max_sims = sims.max(axis=1)                 # (N,)

    # 6) Render
    if boxes is not None:
        for i, box in enumerate(boxes):
            if box is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            name = names[i]
            conf = float(confs[i])

            label = name
            # probability gate
            if conf < prob_threshold:
                label = "Unknown"
            # similarity gate
            if (max_sims is not None) and (float(max_sims[i]) < sim_threshold):
                label = "Unknown"

            # rectangle + label text (no "best class")
            color = (0, 255, 0) if label != "Unknown" else (0, 200, 255)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 3)

            if label == "Unknown":
                if max_sims is not None:
                    text = f"Unknown (p={conf:.2f}, sim={float(max_sims[i]):.2f})"
                else:
                    text = f"Unknown (p={conf:.2f})"
            else:
                if max_sims is not None:
                    text = f"{name} (p={conf:.2f}, sim={float(max_sims[i]):.2f})"
                else:
                    text = f"{name} (p={conf:.2f})"

            draw_label(bgr, x1, y1, text, color=color)

            # results row (no best_name)
            row = {"box": [x1, y1, x2, y2], "label": label, "confidence": conf}
            if max_sims is not None:
                row["cosine_sim"] = float(max_sims[i])
            results.append(row)
    else:
        # Rare: faces returned but no boxes
        name = names[0]
        conf = float(confs[0])
        label = name if conf >= prob_threshold else "Unknown"
        if (max_sims is not None) and (float(max_sims[0]) < sim_threshold):
            label = "Unknown"
        text = f"{label} (p={conf:.2f})"
        if max_sims is not None:
            text = f"{label} (p={conf:.2f}, sim={float(max_sims[0]):.2f})"
        draw_label(bgr, 10, 40, text)
        row = {"box": None, "label": label, "confidence": conf}
        if max_sims is not None:
            row["cosine_sim"] = float(max_sims[0])
        results.append(row)

    return bgr, results

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Unknown detection settings")
prob_threshold = st.sidebar.slider("Min top-class probability", 0.30, 0.95, 0.60, 0.01)

have_centroids = C is not None
use_distance_check = st.sidebar.checkbox(
    "Enable embedding similarity check (cosine)", value=have_centroids, disabled=not have_centroids,
    help="Requires models/embeddings.npz to compute class centroids."
)
sim_threshold = st.sidebar.slider(
    "Min cosine similarity to any class", 0.10, 0.90, 0.45, 0.01, disabled=not use_distance_check
)

# ----------------------------
# Main UI (upload-only)
# ----------------------------
st.subheader("Upload a photo")
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded is None:
    st.info("Upload a player photo to get a prediction.")
    st.stop()

# Read via OpenCV
bgr = uploaded_file_to_bgr(uploaded)
st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

with st.spinner("Analyzing face(s)..."):
    annotated_bgr, outputs = predict_bgr(
        bgr.copy(),
        prob_threshold=prob_threshold,
        use_distance_check=use_distance_check,
        sim_threshold=sim_threshold
    )

st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), caption="Prediction", use_container_width=True)

if len(outputs) == 0:
    st.warning("No face detected.")
else:
    st.write("### Results")
    for i, r in enumerate(outputs, 1):
        line = f"**Face {i}:** {r['label']} (p=`{r['confidence']:.3f}`)"
        if "cosine_sim" in r:
            line += f", cos sim=`{r['cosine_sim']:.3f}`"
        st.write(line)

