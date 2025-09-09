# Plant Disease Detection Using Deep Learning

# 🌱 Plant Disease Recognition System

This project is a **deep learning-based system** that detects plant diseases from leaf images.  
It uses **Convolutional Neural Networks (CNNs)** to classify plant leaves as *healthy* or *diseased* and further identify the specific disease category.  

---

#### Dataset Link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

---

## 🚀 Features
- **Model Training**  
  - Built with **TensorFlow/Keras**.  
  - Trained on a labeled dataset of plant leaf images.  
  - Achieves high accuracy on both training and validation sets.  

- **Model Testing**  
  - Accepts an image of a plant leaf.  
  - Predicts the disease type with confidence score.  

- **Notebooks Included**  
  - `Train_plant_disease.ipynb` → Training pipeline (data preprocessing, CNN model, training, evaluation, saving the model).  
  - `Test_Plant_Disease.ipynb` → Testing pipeline (loading model, predicting disease from new images).  

---

## 🛠️ Tech Stack
- **Python**  
- **TensorFlow / Keras**  
- **NumPy, Pandas**  
- **Matplotlib, Seaborn**  
- **Jupyter Notebook**  

---

## 📂 Workflow
1. **Data Preprocessing**  
   - Images resized and normalized.  
   - Dataset split into training, validation, and test sets.  

2. **Model Building**  
   - CNN designed for multi-class classification.  
   - Optimizer: `Adam` / `RMSProp`.  
   - Loss function: `Categorical Crossentropy`.  

3. **Training & Evaluation**  
   - Trained for multiple epochs.  
   - Performance evaluated with accuracy/loss curves and confusion matrix.  

4. **Testing / Inference**  
   - Load trained model.  
   - Input any leaf image.  
   - Get the predicted plant disease label + confidence score.  

---

## 📊 Results
- High accuracy on both training and validation datasets.
- Training accuracy = 99.7026%
- Validation accuracy = 97.65%
- Potential for real-world agricultural applications.  

---

## 🔮 Future Scope
- Deploy with **Streamlit** for an interactive web app.  
- Expand dataset for more crop species.  
- Optimize for **Edge AI / mobile deployment**.  


⭐ If you like this project, don’t forget to **star the repo** on GitHub!
