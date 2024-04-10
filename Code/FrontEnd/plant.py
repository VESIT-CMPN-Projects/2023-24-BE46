from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import base64

from io import BytesIO

app = Flask(__name__)

# Load the pre-trained MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(11, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the weights of the pre-trained model
model.load_weights('/Users/vanshtakrani/Downloads/Desktop/BE Project Final/Front End/Plant Identification/best_model.h5')  # Update with your actual model file name

def detect_spots(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a minimum area threshold to filter small contours (adjust as needed)
    min_area = 50

    # Iterate over detected contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        if area > min_area:
            # Get the coordinates of the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw a green square around the detected spot
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the image with detected spots
    detection_result_path = "static/detection_result.jpg"
    cv2.imwrite(detection_result_path, image)

    return detection_result_path


@app.route('/')
def home():
    return render_template('home.html', active_page='home')

@app.route('/plant')
def plant():
    return render_template('plant.html', active_page='plant')

@app.route('/pest')
def pest():
    return render_template('pest.html', active_page='pest')

@app.route('/fertilizers')
def fertilizer():
    return render_template('fertilizers.html', active_page='fertilizer')

def make_prediction(image_path):
    img = load_img(image_path, target_size=(224, 224))
    i = img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im, axis=0)
    predicted_probabilities = model.predict(img)[0]

    pred_index = np.argmax(predicted_probabilities)
    class_names = [
        'Pepper__bell___Bacterial_spot',
        'Pepper__bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato_potato__healthy',
        'Tomato_tomato_Bacterial_spot',
        'Tomato_tomato_Leaf_Mold',
        'Tomato_Spider_mites_Two_spotted_spider_mite',
        'Tomato__Tomato__TargetSpot',
        'Tomato__Tomato_YellowLeaf__Curl_Virus',
        'Tomato_tomato_healthy'
    ]
    identified_class = class_names[pred_index]
    parts = identified_class.split('_')
    plant_name = parts[0].capitalize()
    disease_name = ' '.join([word.capitalize() for word in parts[2:]])

    # Convert the image to base64 encoding
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return {'plant_name': plant_name, 'disease_name': disease_name, 'image': encoded_image}

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'})

        image = request.files['image']
        img_path = "temp.jpg"
        image.save(img_path)

        prediction_result = make_prediction(img_path)
        detection_result_path = detect_spots(img_path)

        os.remove(img_path)

        return jsonify({'prediction_result': prediction_result, 'detection_result_path': detection_result_path})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



import joblib
import pandas as pd
model1 = joblib.load("/Users/vanshtakrani/Downloads/Desktop/BE Project Final/Front End/Plant Identification/fertilizer_prediction_model_SVM.joblib")


# Assuming you have a DataFrame containing fertilizer names and their image URLs
fertilizer_data = pd.read_csv("/Users/vanshtakrani/Downloads/Desktop/BE Project Final/Front End/Plant Identification/Fertilizer Prediction_final.csv")

@app.route('/fertilizers', methods=['POST'])
def fertilizers():
    data = request.form.to_dict()
    user_input = pd.DataFrame({
        "Temperature": [int(data["temperature"])],
        "Humidity": [int(data["humidity"])],
        "Moisture": [int(data["moisture"])],
        "Soil Type": [data["soilType"]],
        "Nitrogen": [int(data["nitrogen"])],
        "Potassium": [int(data["potassium"])],
        "Phosphorous": [int(data["phosphorous"])]
    })

    # Predict the fertilizer name
    predicted_fertilizer = model1.predict(user_input)[0]
    print("Predicted Fertilizer:", predicted_fertilizer)  # Debugging statement

    # Get the image URL corresponding to the predicted fertilizer
    fertilizer_image_url = fertilizer_data[fertilizer_data['Fertilizer Name'] == predicted_fertilizer]['Img url'].values[0]
    print("Fertilizer Image URL:", fertilizer_image_url)  # Debugging statement

    # Return the predicted fertilizer name and its image URL
    return jsonify({"predicted_fertilizer": predicted_fertilizer, "image_url": fertilizer_image_url})


if __name__ == '__main__':
    app.run(debug=True,port=5000)
