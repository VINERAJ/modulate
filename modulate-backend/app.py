from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import numpy as np
import os
from werkzeug.utils import secure_filename
from scipy.io import wavfile
from scipy.signal import resample_poly
import pandas as pd

app = Flask(__name__)
CORS(app, resources={
    r"/analyze": {"origins": "*", "methods": ["POST", "OPTIONS"]},
    r"/songs": {"origins": "*", "methods": ["POST", "OPTIONS"]},
    r"/health": {"origins": "*", "methods": ["GET", "OPTIONS"]}
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model and feature extractor
print("Loading model... (this may take a minute)")
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = AutoModelForAudioClassification.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Emotion mapping
id2label = {
    "0": "angry", "1": "calm", "2": "disgust", "3": "fearful",
    "4": "happy", "5": "neutral", "6": "sad", "7": "surprised"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_wav_mono_16k(audio_file_path):
    sample_rate, audio = wavfile.read(audio_file_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)

    target_sr = 16000
    if sample_rate != target_sr:
        audio = resample_poly(audio, target_sr, sample_rate).astype(np.float32)

    return audio

def predict_mood_result(audio_file_path):
    """
    Predict the emotion from an audio file.
    Args:
        audio_file_path: Path to the audio file
    Returns:
        Emotion label as string
    """
    try:
        sound_array = load_wav_mono_16k(audio_file_path)

        input_values = feature_extractor(
            raw_speech=sound_array,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(input_values.input_values.float()).logits
            
        predicted_id = torch.argmax(logits, dim=-1).item()
        emotion = id2label[str(predicted_id)]
        
        # Get confidence score
        probabilities = F.softmax(logits, dim=-1)
        confidence = probabilities[0, predicted_id].item()
        
        return emotion, confidence
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")
    
songs_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'songs.csv'))

def get_song_for_mood(mood):
    mood = mood.strip().lower()

    # Map model emotions to uplifting/appropriate CSV labels: calm, fearful, happy, sad
    emotion_map = {
        "angry": "calm",       # calm down angry users
        "disgust": "happy",    # uplift negative feelings
        "neutral": "happy",    # energize neutral mood
        "surprised": "happy",  # match high energy
        "sad": "happy",        # uplift sad users
        "fearful": "calm",     # soothe fearful users
    }
    mapped_mood = emotion_map.get(mood, mood)

    # Try mapped mood match first
    mood_matches = songs_df[
        songs_df["labels"].fillna("").str.lower().str.contains(mapped_mood, na=False)
    ]

    if not mood_matches.empty:
        n = min(5, len(mood_matches))
        selected = mood_matches.sample(n=n)
    elif len(songs_df) > 0:
        n = min(5, len(songs_df))
        selected = songs_df.sample(n=n)
    else:
        return []

    songs = []
    for _, row in selected.iterrows():
        spotify_url = str(row.get("spotify_url", ""))
        # Extract Spotify track ID from URL
        spotify_id = None
        if "track/" in spotify_url:
            spotify_id = spotify_url.split("track/")[-1].split("?")[0].strip()
        songs.append({
            "artist": str(row.get("artist", "Unknown")),
            "title": str(row.get("title", "Unknown")),
            "spotify_id": spotify_id,
            "spotify_url": spotify_url,
        })
    return songs
    
@app.route('/')
def home():
    return "Welcome!"

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_audio():
    """
    Endpoint to analyze audio emotion.
    Expects a multipart form-data POST request with an 'audio' file.
    Returns JSON with emotion prediction and confidence score.
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use wav'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict emotion
        emotion, confidence = predict_mood_result(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence, 4),
            'success': True
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500
    
@app.route('/songs', methods=['POST', 'OPTIONS'])
def get_songs():
    """
    Endpoint to get songs based on emotion.
    Expects JSON body with 'emotion' field.
    Returns JSON with list of songs matching the emotion.
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        if not data or 'emotion' not in data:
            return jsonify({'error': 'No emotion provided'}), 400
        
        emotion = data['emotion'].lower()
        
        songs_list = get_song_for_mood(emotion)
        print(songs_list)
        
        return jsonify({
            'songs': songs_list,
            'success': True
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)

