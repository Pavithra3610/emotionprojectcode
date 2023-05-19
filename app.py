from flask import Flask, render_template, Response, request
from camera import VideoCamera
import librosa
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# Define the prediction route
# Open the PKL file in binary mode
with open('mlp_model.pkl', 'rb') as f:
    # Load the object from the file
    model = pickle.load(f)

# Define the emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
@app.route('/predict', methods=['POST'])
def predict():
    # Get the audio file from the form
    audio_file = request.files['audio_file']
    # Extract features from the audio file
    X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
    result = np.array([])
    stft = np.abs(librosa.stft(X))
    chromas = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chromas))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128).T, axis=0)
    result = np.hstack((result, mels))
    # Make a prediction using the trained model
    X_test = result.reshape(1, -1)
    prediction = model.predict(X_test)
    # Convert predicted emotion string to its index in the emotions list
    predicted_index = emotions.index(prediction[0])
    # Print the predicted emotion
    predicted_emotion = emotions[predicted_index]
    return render_template('result.html', predicted_emotion=predicted_emotion)
if __name__ == '__main__':
   # app.run(host='localhost', debug=True)
   app.debug = True
   app.run(host="0.0.0.0", port=8300)
