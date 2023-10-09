from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json
import numpy as np
import os
import textwrap
import librosa
import assemblyai as aai
from werkzeug.utils import secure_filename

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
app = Flask(__name__)

aai.settings.api_key = "1b50de60dce8418da258c93bd4f92296"

# Directory for storing uploaded audio files.
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    return getJson(emotions[index], y_chunk_model1.flatten())

def getJson(sent, predicted_proba):
    # Convert the predicted_proba array to a Python list and round each value to two decimal places
    predicted_proba_list = [round(prob, 2) for prob in predicted_proba.tolist()]

    return {
        "score_neutral": predicted_proba_list[0],
        "score_calm": predicted_proba_list[1],
        "score_happy": predicted_proba_list[2],
        "score_sad": predicted_proba_list[3],
        "score_angry": predicted_proba_list[4],
        "score_fearful": predicted_proba_list[5],
        "score_disgust": predicted_proba_list[6],
        "score_surprised": predicted_proba_list[7],
        "prominent_sentiment": sent
    }

# Route for the home page.
@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":    
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            audio_sentiment_model = load_audio_sentiment_model()
            try:
                emotion = predict_sentiment_from_audio(file_path, audio_sentiment_model)
                jsonObject = send_to_api(file_path)
                print(textwrap.fill(str(jsonObject), 100))
                return jsonify(emotion)
            
            except Exception as e:
                return jsonify({"error": str(e)})
            
    return jsonify({"error": "Invalid request"})

# Route for the login page
@app.route('/login', methods=["GET"])
def login():
    if request.method == "GET":
        return render_template('coming-soon.html')

################################ Assembly AI API ###########################################
def send_to_api(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path, aai.TranscriptionConfig(sentiment_analysis=True))
    sentiment = analyze_sentiment(transcript)
    #print(transcript.text)
    return {
        "sentiment": sentiment,
    }

def analyze_sentiment(transcript):
    all_sentiment_scores = [(sentiment.sentiment) for sentiment in transcript.sentiment_analysis]
    sentiments_count = {"POSTIVE": all_sentiment_scores.count("POSITIVE"), "NEGATIVE": all_sentiment_scores.count("NEGATIVE")}
    if sentiments_count["POSTIVE"] > sentiments_count["NEGATIVE"]:
        return "POSITVE"
    elif sentiments_count["POSTIVE"] < sentiments_count["NEGATIVE"]:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

# config = aai.TranscriptionConfig(auto_highlights=True)
# transcript = transcriber.transcribe(
#     FILE_URL,
#     config=config
# )

# for result in transcript.auto_highlights.results:
#     print(f"Highlight: {result.text}, Count: {result.count}, Rank: {result.rank}")


if __name__ == '__main__':
    app.run(debug=True)
