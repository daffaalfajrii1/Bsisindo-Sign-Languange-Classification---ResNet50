import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('bisindo_resnet50.h5')

# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('bisindo_resnet_model.tflite', 'wb') as f:
    f.write(tflite_model)
