🧠 Medical Image Classification using Deep Learning
📌 Project Overview

This project focuses on the development of a deep learning model for the classification of medical images (Chest X-rays) into normal and abnormal cases.
The objective is to explore the application of computer vision techniques in a medical context using limited data.

🎯 Objectives
Apply deep learning (CNN) to medical image analysis
Evaluate model performance on unseen data
Understand generalization and overfitting in small datasets
🧠 Methodology
🔹 Model
Transfer Learning using ResNet18 (pretrained on ImageNet)
Fine-tuned for binary classification (normal vs abnormal)
🔹 Data
Subset of Chest X-ray dataset
Organized into training and validation sets
🔹 Training
Loss function: CrossEntropyLoss
Optimizer: Adam
Input size: 224x224
📊 Results
Training Loss decreased significantly over epochs
Validation Accuracy: ~87.5%

This indicates good generalization despite the small dataset size.

📈 Evaluation
Confusion Matrix

Used to analyze classification errors and model performance across classes.

Visualization

Sample predictions are visualized to compare model outputs with ground truth labels.

⚠️ Limitations
Small dataset size
Potential overfitting due to limited data
No advanced preprocessing or augmentation applied
🚀 Future Improvements
Data augmentation techniques
Medical image segmentation (e.g., U-Net)
3D medical imaging analysis (CT scans)
Image registration approaches
🛠️ Technologies Used
Python
PyTorch
Torchvision
Scikit-learn
Matplotlib
📁 Project Structure
medical-image-classification/
├── data/
├── models/
├── src/
├── results/
├── README.md
💡 Key Takeaways

This project demonstrates the ability to:

Work with deep learning models for image data
Handle medical datasets
Evaluate model performance in a research-oriented setting
👤 Author

Mohammed Fassi Fehri