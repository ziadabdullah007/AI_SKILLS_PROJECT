<div align="center">

# Face Recognition System (Clustering + Recognition)
### Compare 3 Models • Pick Best • Deploy in GUI

<img src="https://raw.githubusercontent.com/ziadabdullah07/Face-Recognition-System/main/docs/demo.gif" width="100%"/>

**Face Recognition & Smart Attendance Course Project**  
Trained on LFW dataset using transfer learning, model comparison & Grad-CAM, deployed via GUI for top-3 identity prediction.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)](https://streamlit.io)

</div>

---

## 🎯 Project Objective

Build a face recognition system that:

1. **Loads & preprocesses LFW images**
2. **Extracts face embeddings using 3 CNN models (Transfer Learning)**
3. **Evaluates & compares models using**:
   - Accuracy
   - Confusion Matrix
   - Grad-CAM attention maps
4. **Selects the best model based on accuracy**
5. **Deploys the best model into a GUI that supports**:
   - Image upload
   - Live webcam recognition
   - Top-3 identity prediction with confidence scores
   - Grad-CAM visualization
   - Accuracy / Confusion Matrix display

---

## 🧠 Models Used in Comparison

| Task | Models |
|------|--------|
| Face embeddings + transfer learning | ✅ ResNet50 , ✅ InceptionV3 , ✅ MobileNetV2 |
| Similarity metric | Cosine Similarity + Euclidean (L2 distance) |
| Face detection (optional step if GUI uses webcam) | MTCNN or RetinaFace |

> After evaluation, **the best model is selected and saved**, then loaded by the GUI.




