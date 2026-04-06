# 🧠 Medical Image Classification using Deep Learning

## 📌 Project Overview

This project focuses on the development of a deep learning model for the classification of medical images (Chest X-rays) into normal and abnormal cases.
The objective is to explore the application of computer vision techniques in a medical context using limited data.

---

## 🎯 Objectives

* Apply deep learning (CNN) to medical image analysis
* Evaluate model performance on unseen data
* Understand generalization and overfitting in small datasets

---

## 🧠 Methodology

### 🔹 Model

* Transfer Learning using ResNet18 (pretrained on ImageNet)
* Fine-tuned for binary classification (normal vs abnormal)

### 🔹 Data

* Subset of Chest X-ray dataset
* Organized into training and validation sets

### 🔹 Training

* Loss function: CrossEntropyLoss
* Optimizer: Adam
* Input size: 224x224

---

## 📊 Results

* Training Loss decreased significantly over epochs
* Validation Accuracy: **~87.5%**

This indicates good generalization despite the small dataset size.

---

## 📈 Evaluation

### Confusion Matrix

Used to analyze classification errors and model performance across classes.

### Visualization

Sample predictions are visualized to compare model outputs with ground truth labels.

---

## ⚠️ Limitations

* Small dataset size
* Potential overfitting due to limited data
* No advanced preprocessing or augmentation applied

---

## 🚀 Future Improvements

* Data augmentation techniques
* Medical image segmentation (e.g., U-Net)
* 3D medical imaging analysis (CT scans)
* Image registration approaches

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* Scikit-learn
* Matplotlib

---

## 📁 Project Structure

```
medical-image-classification/
├── data/
├── models/
├── src/
├── results/
├── README.md
```

---

## 💡 Key Takeaways

This project demonstrates the ability to:

* Work with deep learning models for image data
* Handle medical datasets
* Evaluate model performance in a research-oriented setting

---

## 👤 Author

Mohammed Fassi Fehri

## 📌 Project Summary

### 🇬🇧 English

This project explores the application of deep learning techniques to medical image classification, specifically chest X-ray images.
A convolutional neural network (ResNet18 pretrained on ImageNet) was used within a transfer learning framework to handle limited data.

The model was trained on a small dataset and evaluated on a separate validation set, achieving approximately 87.5% validation accuracy.
This result indicates a reasonable generalization capability despite the limited size of the dataset.

The project highlights key steps of a computer vision pipeline in a medical context, including image preprocessing, supervised learning, model evaluation, and result visualization.

---

### 🇫🇷 Français

Ce projet explore l’application de techniques de deep learning à la classification d’images médicales (radiographies pulmonaires).
Un modèle de type CNN (ResNet18 pré-entraîné sur ImageNet) a été utilisé dans une approche de transfer learning afin de s’adapter à un contexte de données limitées.

Le modèle a été entraîné sur un petit jeu de données, puis évalué sur un ensemble de validation indépendant, atteignant une précision d’environ 87,5 %, ce qui montre une capacité de généralisation satisfaisante.

Ce projet met en évidence les étapes clés d’un pipeline de vision par ordinateur en contexte médical, incluant le prétraitement des images, l'apprentissage supervisé, l'évaluation des performances et la visualisation des résultats.
