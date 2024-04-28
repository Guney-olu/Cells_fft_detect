import cv2
import numpy as np
import tensorflow as tf

# for directly give fft applied images
model = tf.keras.models.load_model('Models/cell_classifier_model_with_augmentation_update.h5') #change the model here
model.compile(optimizer=model.optimizer, loss=model.loss, metrics=['accuracy'])

class_names = ['eosinophil','monocytes', 'platelet']

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (360, 360)) 
    img = img.astype(np.float32) / 255.0 
    img = np.expand_dims(img, axis=0) 
    return img

def predict_class(img_path):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    return predicted_class

def get_class_name(class_index):
    return class_names[class_index]

image_path = 'cells_dataset/eosinophil/EO_442.jpg'  
predicted_class = predict_class(image_path)
predicted_class_name = get_class_name(predicted_class)
print('Predicted class:', predicted_class_name)
