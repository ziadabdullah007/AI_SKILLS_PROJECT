import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time

# ---------------------------
# SETTINGS - Models
# ---------------------------
MODELS_CONFIG = {
    "VGG16": {
        "path": r"C:\Users\Ahmed\Downloads\ahmed\Models\meena_V1_c50_s50_acc93.h5",
        "image_size": (224, 224),
        "last_conv_layer": None,
        "framework": "keras",
        "preprocessing": "vgg"
    },
}

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_all_models():
    models = {}
    for name, config in MODELS_CONFIG.items():
        try:
            if config["framework"] == "keras":
                models[name] = load_model(config["path"])
                st.success(f"✅ تم تحميل {name} (Keras)")
        except Exception as e:
            st.error(f"❌ فشل تحميل {name}: {str(e)}")
            models[name] = None
    return models

models = load_all_models()

# ---------------------------
# Preprocessing Functions
# ---------------------------
def preprocess_vgg(frame, image_size):
    """VGG16 preprocessing: BGR + mean subtraction"""
    img = cv2.resize(frame, image_size)
    img = img.astype("float32")
    # VGG mean values (BGR format)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_standard(frame, image_size):
    """Standard preprocessing: scale to [0, 1]"""
    img = cv2.resize(frame, image_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------------------
# Overlay Heatmap
# ---------------------------
def overlay_heatmap(img, heatmap, alpha=0.28, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(img, 0.74, heatmap_color, alpha, 0)
    return overlay

# ---------------------------
# Prediction - Keras
# ---------------------------
def predict_top3_keras(frame, model, config):
    preprocessing_type = config.get("preprocessing", "standard")
    
    if preprocessing_type == "vgg":
        img = preprocess_vgg(frame, config["image_size"])
    else:
        img = preprocess_standard(frame, config["image_size"])
    
    preds = model.predict(img, verbose=0)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    results = [(f"Class {i}", float(preds[i])) for i in top3_idx]
    
    return results, img, preds

# ---------------------------
# Grad-CAM - Keras
# ---------------------------
def make_gradcam_keras(img_array, model, class_index, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        return np.random.rand(7, 7)
    
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if isinstance(predictions, list):
                predictions = predictions[0]
            loss = predictions[:, class_index]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= (tf.reduce_max(heatmap) + 1e-8)
        
        return heatmap.numpy()
    except Exception as e:
        st.warning(f"⚠️ Grad-CAM خطأ في Keras: {str(e)}")
        return np.random.rand(7, 7)

# ---------------------------
# Draw label
# ---------------------------
def draw_label(frame, text, position=(10, 30)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🔸 Face Recognition + Grad-CAM")
st.write("✅ نظام التعرف على الوجوه مع Grad-CAM")

fps_limit = st.slider("⚙️ عدد الفريمات في الثانية (FPS)", 1, 30, 10)
update_interval = 1.0 / fps_limit

run = st.checkbox("تشغيل الكاميرا")
show_gradcam = st.checkbox("تفعيل Grad-CAM Overlay")

frame_area = st.empty()
top3_area = st.empty()

# إدارة الكاميرا
if 'cap' not in st.session_state:
    st.session_state.cap = None

if run:
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap.isOpened():
            st.error("❌ فشل فتح الكاميرا")
            st.stop()
    
    cap = st.session_state.cap
    last_update = time.time()
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ فشل قراءة الفريم من الكاميرا")
            break
        
        frame = cv2.flip(frame, 1)
        current_time = time.time()
        
        if current_time - last_update >= update_interval:
            last_update = current_time
            
            model_name = "VGG16"
            config = MODELS_CONFIG[model_name]
            model = models[model_name]
            
            if model is not None:
                try:
                    # Predict
                    top3, img_array, preds = predict_top3_keras(frame, model, config)
                    
                    if show_gradcam:
                        class_index = np.argmax(preds)
                        heatmap = make_gradcam_keras(img_array, model, class_index, config["last_conv_layer"])
                        frame_with_cam = overlay_heatmap(frame.copy(), heatmap)
                    else:
                        frame_with_cam = frame.copy()
                    
                    # Draw & Display
                    result_img = draw_label(frame_with_cam, f"{top3[0][0]} ({top3[0][1]*100:.1f}%)")
                    frame_area.image(result_img[:, :, ::-1], channels="RGB", use_container_width=True)
                    
                    with top3_area.container():
                        st.subheader(f"🏆 Top 3 Predictions")
                        for label, acc in top3:
                            st.write(f"**{label}** – {acc*100:.2f}%")
                            
                except Exception as e:
                    st.error(f"❌ خطأ: {str(e)}")
        
        if not st.session_state.get('run', True):
            break
else:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        st.info("📹 تم إيقاف الكاميرا")