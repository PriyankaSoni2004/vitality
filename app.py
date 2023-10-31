from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json
import numpy as np
import os
import textwrap
import librosa
import pickle
import assemblyai as aai
from werkzeug.utils import secure_filename

emotions = ['neutral', 'sad', 'happy', 'angry', 'disgust', 'fear', 'surprise']
app = Flask(__name__)

aai.settings.api_key = "1b50de60dce8418da258c93bd4f92296"

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open('model/scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('model/encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

def load_audio_sentiment_model():
    model_architecture_path = 'model/CNN_model.json'
    model_weights_path = 'model/CNN_model_weights.h5'    
    with open(model_architecture_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    audio_sentiment_model = model_from_json(loaded_model_json)
    audio_sentiment_model.load_weights(model_weights_path)
    
    return audio_sentiment_model

# Function to preprocess audio file and predict sentiment.
# def predict_sentiment_from_audio(audio_file_path, model):

    # X, sr = librosa.load(audio_file_path, sr=None)
    # stft = np.abs(librosa.stft(X))
    # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)
    # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    # mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sr).T, axis=0)
    # contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)
    # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)
    # features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    # feature_all = np.vstack([features])

    # x_chunk = np.array(features)
    # x_chunk = x_chunk.reshape(1, np.shape(x_chunk)[0])
    # y_chunk_model1 = model.predict(x_chunk)
    # index = np.argmax(y_chunk_model1)
    # print(emotions[index])
    # return getJson(emotions[index], y_chunk_model1.flatten())

#################################### PREPROCESSING ###################################################
def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data,frame_length,hop_length):
    rmse=librosa.feature.rms(y = data ,frame_length=2048,hop_length=512)
    return np.squeeze(rmse)

def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y = data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,2048,512),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result


def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    i_result = scaler2.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    return final_result

emotions1={0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

def prediction(path1, model):
    res=get_predict_feat(path1)
    predictions = model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    return getJson(y_pred[0][0], predictions)

def getJson(sent, predicted_proba):
    # Round the elements of the predicted_proba NumPy array to two decimal places
    predicted_proba_list = np.round(predicted_proba, 2).tolist()
    predicted_proba_list = predicted_proba_list[0]
    print(predicted_proba_list)
    if len(predicted_proba_list) >= 7:
        return {
            "score_angry": predicted_proba_list[0],
            "score_disgust": predicted_proba_list[1],
            "score_fear": predicted_proba_list[2],
            "score_happy": predicted_proba_list[3],
            "score_neutral": predicted_proba_list[4],
            "score_sad": predicted_proba_list[5],
            "score_surprise": predicted_proba_list[6],
            "prominent_sentiment": sent
        }
    else:
        return {
            "error": "Not enough elements in predicted_proba_list"
        }

#################################### ROUTES ######################################################
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
                emotion = prediction(file_path, audio_sentiment_model)
                # jsonObject = send_to_api(file_path)
                # print(textwrap.fill(str(jsonObject), 100))
                return jsonify(emotion)
            except Exception as e:
                return jsonify({"error": str(e)})
            
    return jsonify({"error": "Invalid request"})

# Route for the login page
@app.route('/login', methods=["GET"])
def login():
    if request.method == "GET":
        return render_template('login.html')

################################ Assembly AI API ###########################################
def send_to_api(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path, aai.TranscriptionConfig(sentiment_analysis=True))
    sentiment = analyze_sentiment(transcript)
    print(transcript.text)
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
