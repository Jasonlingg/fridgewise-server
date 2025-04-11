import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow_datasets as tfds

model = tf.keras.models.load_model('food101_model.h5')
print("Model loaded from 'food101_model.h5'")

_, info = tfds.load('food101', with_info=True)

def predict_burger(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_names = info.features['label'].names
    predicted_label = class_names[predicted_class]
    
    return predicted_label

# Test with a burger image
img_path = 'Hamburger.jpeg' 
predicted_label = predict_burger(img_path)
print(f"Predicted label: {predicted_label}")