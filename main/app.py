# Importing necessary libraries.
from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json
import numpy as np
import os
import librosa
from werkzeug.utils import secure_filename

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Creating a Flask app.
app = Flask(__name__)

# Directory for storing uploaded audio files.
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pre-trained audio sentiment analysis model.
def load_audio_sentiment_model():
    model_architecture_path = 'model/mlp_model_tanh_adadelta.json'
    model_weights_path = 'model/mlp_tanh_adadelta_model.h5'
    
    with open(model_architecture_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    
    audio_sentiment_model = model_from_json(loaded_model_json)
    audio_sentiment_model.load_weights(model_weights_path)
    
    return audio_sentiment_model

# Function to preprocess audio file and predict sentiment.
def predict_sentiment_from_audio(audio_file_path, model):

    X, sr = librosa.load(audio_file_path, sr=None)
    stft = np.abs(librosa.stft(X))

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sr).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

    feature_all = np.vstack([features])

    x_chunk = np.array(features)
    x_chunk = x_chunk.reshape(1, np.shape(x_chunk)[0])
    y_chunk_model1 = model.predict(x_chunk)
    index = np.argmax(y_chunk_model1)
    
    print(emotions[index])
    return emotions[index]
    sentiment_scores = {
        "val_neg": 0.2,
        "val_neu": 0.6,
        "val_pos": 0.2
    }
    #return sentiment_scores

# Route for the home page.
@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":    
        file = request.files.get("input")
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            audio_sentiment_model = load_audio_sentiment_model()

            try:
                # Perform sentiment analysis on the uploaded audio file.
                sentiment = predict_sentiment_from_audio(file_path, audio_sentiment_model)
                print(sentiment)
                return "<p>sentiment</p>"
                #return jsonify({"sentiment": sentiment})

            except Exception as e:
                return jsonify({"error": str(e)})
            
    return jsonify({"error": "Invalid request"})

if __name__ == '__main__':
    app.run()
