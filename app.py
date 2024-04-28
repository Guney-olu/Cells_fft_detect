from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

class_names = ['eosinophil','monocytes', 'platelet']

model = tf.keras.models.load_model('Models/cell_classifier_model_with_augmentation_update.h5')

model.make_predict_function()

def filter_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Failed to load image"
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.float64)

    Y = np.fft.fftshift(np.fft.fft2(gray_image))

    rows, columns = Y.shape
    B = Y.copy()
    centerX = columns // 2
    centerY = rows // 2
    filterWidth = 15
    B[centerY - filterWidth:centerY + filterWidth, centerX - filterWidth:centerX + filterWidth] = 0

    # Inverse FFT
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(B)))
    
    return filtered_image, None

def preprocess_image(filtered_image):
    img = cv2.imread(filtered_image, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, (360, 360)) 
    img = img.astype(np.float32) / 255.0 
    img = np.expand_dims(img, axis=0) 
    return img

def predict_class(img):
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    pred = class_names[predicted_class]
    return pred


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("upload_file.html")

@app.route("/about")
def about_page():
    return "Kid Stuff"

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename 
        img.save(img_path)
        print("we are printing image path", img_path)
        
        filtered_img, _ = filter_image(image_path=img_path)
        filtered_img_path = "static/filtered_" + os.path.basename(img_path)  
        cv2.imwrite(filtered_img_path, filtered_img) 
        
        print("after filtering", filtered_img_path)
        
        preprocessed_img = preprocess_image(filtered_image=filtered_img_path)
        print("after preprocessing", preprocessed_img)
        
        p = predict_class(preprocessed_img)

    return render_template("upload_file.html", prediction=p, img_path=img_path,filtered_img_path=filtered_img_path)

if __name__ =='__main__':
    #app.debug = True
    app.run(debug=True)
