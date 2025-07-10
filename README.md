# mango-disease-detection
# 🍃 Deep Learning-Driven Mango Disease Diagnosis

An advanced mango disease detection system leveraging **YOLOv5** and **multispectral UAV imagery** for real-time, high-accuracy classification and localization of crop diseases. The project introduces **optimizer blending** (SGD + AdamW) and is tailored for deployment in **precision agriculture**, empowering early disease intervention directly from aerial drone footage.

---

## 🚀 Features

- YOLOv5 object detection for mango leaf diseases
- Integrated Multispectral band support for better disease contrast
- Optimizer blending: SGD + AdamW for enhanced convergence
- UAV drone field data used for realistic performance
- Designed for edge deployment (Jetson Orin Nano, Raspberry Pi, etc.)
- Real-time detection and alerts

---

## 🛠️ Technologies Used

- Python  
- YOLOv5  
- OpenCV  
- Multispectral Image Analysis  
- UAV Drone Imagery  
- SGD + AdamW (Blended Optimizers)

---

## 📁 Dataset

- Custom mango leaf disease dataset captured by drones
- Annotated using Roboflow in YOLO format
- Contains multiple disease classes and healthy samples

---

## 🧪 Training & Evaluation

```bash
# Train the model
python train.py --img 640 --batch 16 --epochs 100 --data mango.yaml --weights yolov5s.pt --optimizer AdamW

# Evaluate performance
python val.py --data mango.yaml --weights runs/train/exp/weights/best.pt
