    import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model  = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Detection"])

#Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE DETECTION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Detection System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Detection** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Detection** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
""")

#About Page
elif(app_mode=="About"):
    st.markdown("## About the Dataset")
    st.markdown("""
    The dataset used in this project is a curated and augmented version of the publicly available PlantVillage dataset from Kaggle. It has been enhanced through offline data augmentation techniques such as rotation, flipping, zooming, and color transformations. These augmentations help increase dataset diversity and improve the model’s ability to generalize.

    This version of the dataset contains approximately **87,000 RGB images** of healthy and diseased crop leaves, categorized into **38 distinct classes**. These classes cover a range of common plant diseases—bacterial, viral, and fungal—as well as healthy samples.

    To ensure effective training and validation:
    - **Training Set**: 70,295 images (80%)
    - **Validation Set**: 17,572 images (20%)
    - **Test Set**: 33 images (manually selected for final evaluation)

    The images are organized to preserve their respective class structure, ensuring clarity and consistency throughout the model's learning process.
    """)

    st.markdown("## Project Members")
    st.markdown("""
    This project was developed by final-year undergraduate students from the **Department of Computer Science and Engineering, JSS Academy of Technical Education**:

    - **Akul Gaur** (Roll No.: 2100910100025)  
    - **Anshika Mittal** (Roll No.: 2100910100034)  
    - **Anushka Sharma** (Roll No.: 2100910100039)

    Each member contributed to various components of the project, including data preprocessing, model architecture, training, evaluation, and deployment.
    """)


    
#Prediction Page
elif(app_mode=="Disease Detection"):
    st.header("Disease Detection")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_container_width=True)
    #Predict Button
    if(st.button("Predict")):
        with st.spinner("Please Wait.."):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Define Class
            class_name = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
