from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import base64

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
from keras.applications import DenseNet121
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained MobileNet model for classification
# Load the pre-trained MobileNet model for classification
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)
classification_model = Model(inputs=base_model.input, outputs=predictions)
classification_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classification_model.load_weights("/Users/vanshtakrani/Downloads/Desktop/BE Project Final/Front End/Plant Identification/best_densenet_model.h5")

# Load the pre-trained Faster R-CNN model for object detection
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
object_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
object_detection_model.eval()

def detect_spots(image_path, pest_name):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to a PyTorch tensor
    image_tensor = transforms.ToTensor()(image).to(device)
    # Perform inference using the Faster R-CNN model
    with torch.no_grad():
        prediction = object_detection_model([image_tensor])
    # Find the largest bounding box
    max_area = 0
    max_box = None
    for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > max_area and score >0.25  :  # Adjust the score threshold as needed
            max_area = area
            max_box = box.int().tolist()
    # Draw the bounding box around the largest detected object (pest) and add text
    if max_box:
        x1, y1, x2, y2 = max_box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add the name of the pest
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
        text = f"{pest_name}"  # Use the provided pest name
        cv2.putText(image, text, (x1, y1 - 10), font, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        # Save the image with the bounding box and text drawn
        detection_result_path = "static/detection_result.jpg"
        cv2.imwrite(detection_result_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return detection_result_path
    else:
        return None



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
    predicted_probabilities = classification_model.predict(img)[0]
    pred_index = np.argmax(predicted_probabilities)
    class_names = [
        'Pepper_pepper_Flea_Beetle',
        'Pepper_Thrips_Thrips',
        'Potato_potato_Black_Cutworm',
        'Potato_potato_Silverleaf_Whitefly',
        'Tomato_tomato_Beet_Armyworm',
        'Tomato_tomato_Red_spider_mite'
    ]
    identified_class = class_names[pred_index]
    parts = identified_class.split('_')
    plant_name = parts[0].capitalize()
    pest_name = ' '.join([word.capitalize() for word in parts[2:]])
    # Convert the image to base64 encoding
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return {'plant_name': plant_name, 'pest_name': pest_name, 'image': encoded_image}

@app.route('/api/predict1', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'})
        image = request.files['image']
        img_path = "temp.jpg"
        image.save(img_path)
        prediction_result = make_prediction(img_path)
        pest_name = prediction_result['pest_name']
        detection_result_path = detect_spots(img_path, pest_name)
        os.remove(img_path)
        return jsonify({'prediction_result': prediction_result, 'detection_result_path': detection_result_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


import pandas as pd 
@app.route('/api/pesticides/<identified_pest>', methods=['GET'])
def get_pesticides(identified_pest):
    try:
        # Assuming 'pesticides_data.csv' contains the pesticide data
        csv_file_path = "/Users/vanshtakrani/Downloads/Desktop/BE Project Final/Front End/Plant Identification/Pesticides.csv"
        df = pd.read_csv(csv_file_path)
        # Filter the dataframe based on the identified pest
        pest_row = df[df['Pest'] == identified_pest]
        if not pest_row.empty:
            # Extract pesticides, images, and URLs
            pesticides = pest_row['Pesticide '].iloc[0]
            image_urls = pest_row['URL'].iloc[0]
            # Split the pesticides and image URLs
            pesticides_list = pesticides.split(', ')
            image_urls_list = image_urls.split(', ')
            # Create a list of dictionaries containing pesticide details
            pesticides_data = [{'name': pesticide, 'image': image_url} for pesticide, image_url in zip(pesticides_list, image_urls_list)]
            return jsonify({'pesticides': pesticides_data})
        else:
            return jsonify({'error': 'Pest not found in the dataset'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


import joblib
import pandas as pd
model = joblib.load("/Users/vanshtakrani/Downloads/Desktop/BE Project Final/Front End/Plant Identification/fertilizer_prediction_model_SVM.joblib")



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
    predicted_fertilizer = model.predict(user_input)[0]
    print("Predicted Fertilizer:", predicted_fertilizer)  # Debugging statement

    # Get the image URL corresponding to the predicted fertilizer
    fertilizer_image_url = fertilizer_data[fertilizer_data['Fertilizer Name'] == predicted_fertilizer]['Img url'].values[0]
    print("Fertilizer Image URL:", fertilizer_image_url)  # Debugging statement

    # Return the predicted fertilizer name and its image URL
    return jsonify({"predicted_fertilizer": predicted_fertilizer, "image_url": fertilizer_image_url})


# import pandas as pd
# from flask import Flask, jsonify, request
# from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder from sklearn.preprocessing
# import h5py
# import numpy as np
# import pickle



# # Load the saved model and label encoder
# model_filename = "/Users/vanshtakrani/Desktop/BE Project/Front End/Plant Identification/pesticide_model.h5"
# label_encoder_filename = "/Users/vanshtakrani/Desktop/BE Project/Front End/Plant Identification/label_encoder.h5"

# # Function to predict pesticides for a given pest
# def predict_pesticides(pest):
#     with h5py.File(model_filename, 'r') as hf:
#         model_path = hf['classifier'][()].astype(str)
    
#     with open(model_path, 'rb') as f:
#         saved_classifier = pickle.load(f)

#     with h5py.File(label_encoder_filename, 'r') as hf:
#         label_classes = hf['classes_'][()]
#         label_encoder = LabelEncoder()
#         label_encoder.classes_ = label_classes

#     pest_encoded = label_encoder.transform([pest])[0]
#     predicted_pesticides = saved_classifier.predict([[pest_encoded]])
#     return predicted_pesticides[0]

# @app.route('/api/pesticides/<identified_pest>', methods=['GET'])
# def get_pesticides(identified_pest):
#     try:
#         # Assuming 'pesticides_data.csv' contains the pesticide data
#         csv_file_path = "/Users/vanshtakrani/Desktop/BE Project/Front End/Plant Identification/Pesticides.csv"
#         df = pd.read_csv(csv_file_path)
#         # Filter the dataframe based on the identified pest
#         pest_row = df[df['Pest'] == identified_pest]
#         print(pest_row)
#         if not pest_row.empty:
#             # Extract pesticides, images, and URLs
#             pesticides = pest_row['Pesticide'].iloc[0]  # Remove the extra space
#             image_urls = pest_row['URL'].iloc[0]        # Remove the extra space
#             # Split the pesticides, URLs
#             pesticides_list = pesticides.split(', ')
#             image_urls_list = image_urls.split(', ')
#             # Predict pesticides using the trained model
#             predicted_pesticides = predict_pesticides(identified_pest)
#             # Create a list of dictionaries containing pesticide details
#             pesticides_data = [{'name': pesticide, 'url': url} for pesticide, url in zip(pesticides_list, image_urls_list) if pesticide in predicted_pesticides]
#             return jsonify({'pesticides': pesticides_data})
#         else:
#             return jsonify({'error': 'Pest not found in the dataset'})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True, port=5001)
