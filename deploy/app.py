import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import mediapipe as mp

app = Flask(__name__)

# Load your Keras model
model_path = 'bisindo_resnet50.h5'  # Change this to the correct path
loaded_model = load_model(model_path)

# Map predicted_class to the corresponding letter or action
category = {
    0: ['a', 'A'], 1: ['b', 'B'], 2: ['c', 'C'],
    3: ['d', 'D'], 4: ['e', 'E'], 5: ['f', 'F'], 6: ['g', 'G'], 7: ['h', 'H'],
    8: ['i', 'U'], 9: ['j', 'J'], 10: ['k', 'K'], 11: ['l', 'L'], 12: ['m', 'M'],
    13: ['n', 'N'], 14: ['o', 'O'], 15: ['p', 'P'], 16: ['q', 'Q'], 17: ['r', 'R'],
    18: ['s', 'S'], 19: ['t', 'T'], 20: ['u', 'U'], 21: ['v', 'V'], 22: ['w', 'W'],
    23: ['x', 'X'], 24: ['y', 'Y'], 25: ['z', 'Z']
}

def predict_image(img_path, model):
    img_ = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)
    index = np.argmax(prediction)

    return category[index][1]

def generate_frames():
    cap = cv2.VideoCapture(0)

    with mp.solutions.hands.Hands() as hands:
        while True:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            cv2.imwrite('static/captured_frame.jpg', frame)
            predicted_letter = predict_image('static/captured_frame.jpg', loaded_model)

            frame = cv2.putText(frame, f'Predicted: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
