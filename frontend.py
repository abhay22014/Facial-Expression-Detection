import cv2
from keras.models import model_from_json
import numpy as np
import streamlit as st
import joblib
from PIL import Image
from streamlit_option_menu import option_menu

# Load the emotion detection model
json_file = open("C:\\Users\\abhaydagar\\OneDrive\\Desktop\\emotion_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("C:\\Users\\abhaydagar\\OneDrive\\Desktop\\emotion_model.h5")

# Load the saved models
decision_tree_model = joblib.load('C:\\Users\\abhaydagar\\OneDrive\\Desktop\\decision_tree_model.h5')
random_forest_model = joblib.load('C:\\Users\\abhaydagar\\OneDrive\\Desktop\\random_forest_model.h5')
adaboost_model = joblib.load('C:\\Users\\abhaydagar\\OneDrive\\Desktop\\adaboost_model.h5')

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from images
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to predict emotion from image
def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    for (p, q, r, s) in faces:
        face_image = gray[q:q+s, p:p+r]
        face_image = cv2.resize(face_image, (48, 48))
        img = extract_features(face_image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        cv2.putText(image, '%s' % prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.rectangle(image, (p, q), (p+r, q+s), (255, 0, 0), 2)
    return image

# Function to preprocess image for models
def preprocess_image_models(image):
    image_gray = image.convert('L')  # Convert image to grayscale
    image_resized = image_gray.resize((10, 10))  # Resize image to target size
    image_array = np.array(image_resized)  # Convert image to numpy array
    image_flat = image_array.flatten()  # Flatten the 2D image array to 1D
    return image_flat.reshape(1, -1)  # Reshape array to match model input shape

# Function to preprocess image for emotion detection model
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (48, 48))  # Resize image to target size
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input shape
    image = np.expand_dims(image, axis=-1)
    image = image / 255.0  # Normalize image
    return image

# Function to predict emotion from image for emotion detection model
def predict_emotion_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    return labels[predicted_label]

# Function to predict emotion from video
def predict_emotion_video(video_path):
    cap = cv2.VideoCapture(video_path)
    st.write(cap)

    if not cap.isOpened():
        st.error("Error: Unable to open video file.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, labels[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()
    cv2.destroyAllWindows()

# Streamlit app
st.title("Emotion Detection")
st.write("This is an Emotion Detection application.")
st.write("You can choose different options to perform emotion detection on images or videos or Via Web Cam")
# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
option="Home"
with st.sidebar:
    option=option_menu(
        menu_title="MAIN MENU",
        options=["Home","Photo (Upload)", "Video (Upload)", "Webcam (Live)","About Us"],

    )
# Define the tabs
tabs = ["Home","Photo (Upload)", "Video (Upload)", "Webcam (Live)","About Us"]
if option==tabs[0]:
    st.title("Our Work")
    st.write("This is an Facial Emotion Detection application.")
    st.write("Our Models Accuracy is around 93% trained using CNN Convolution Neural Networks.")
    st.title("Models")
    st.write("We have used 3 models for prediction of emotions")
    st.write("1. Decision Tree")
    st.write("2. Random Forest")
    st.write("3. Adaboost")
    st.write("4. Connvolution Neural network")
    st.success("The Best Accuracy model is the CNN Model")
    st.title("Developers")
    st.write("This application is developed by Abhay Dagar and Rohan Basugade")
if option ==tabs[1]:
    st.title('Emotion Detection App')
    st.write('Upload an image to detect the emotion')

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect Emotion'):
            image_array = np.array(image)
            images_models = preprocess_image_models(image)

            decision_tree_prediction = decision_tree_model.predict(images_models)
            random_forest_prediction = random_forest_model.predict(images_models)
            adaboost_prediction = adaboost_model.predict(images_models)

            decision_tree_label = labels[decision_tree_prediction[0]]
            random_forest_label = labels[random_forest_prediction[0]]
            adaboost_label = labels[adaboost_prediction[0]]

            st.write('Decision Tree Prediction:', decision_tree_label)
            st.write('Random Forest Prediction:', random_forest_label)
            st.write('AdaBoost Prediction:', adaboost_label)

            emotion_image = predict_emotion_image(image_array)
            st.success(f'Final Predicted Emotion Using CNN (BEST MODEL): {emotion_image}')

elif option == tabs[2]:
    st.title('Emotion Detection App')
    st.write('Upload a video to detect emotions')

    uploaded_video = st.file_uploader("Choose a video...", type=["mp4"])

    if uploaded_video is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(temp_video_path)

        predict_emotion_video(temp_video_path)

elif option == tabs[3]:
    webcam = cv2.VideoCapture(0)

    while True:
        ret, im = webcam.read()
        if not ret:
            break

        im = predict_emotion(im)

        cv2.imshow("Output", im)

        key = cv2.waitKey(1)
        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

elif option==tabs[4]:
    st.title("About Us")
    st.write("""
    The application allows you to detect emotions in images and videos using machine learning models.
    For any inquiries or feedback.\n
    Please contact :\n
    Abhay Dagar  (abhay22014@iiitd.ac.in)\n
    Rohan Basugade (rohan22416@iiitd.ac.in)
    """)
    st.title("Developers")
    st.write("1. Abhay Dagar")
    st.write("2. Rohan Basugade")

