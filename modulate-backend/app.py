from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import numpy as np
import os
from werkzeug.utils import secure_filename
from scipy.io import wavfile
from scipy.signal import resample_poly
import pandas as pd
import modal
from pathlib import Path
from asgiref.wsgi import WsgiToAsgi

# Create Modal app
modal_app = modal.App("modulate-backend")

# Define Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "flask==2.3.3",
        "flask-cors==4.0.0",
        "torch==2.0.1",
        "transformers==4.33.0",
        "numpy==1.24.3",
        "scipy==1.11.2",
        "pandas==2.0.3",
        "werkzeug==2.3.7",
        "asgiref==3.8.1",
    )
)

# Create a volume for temporary file uploads
volume = modal.Volume.from_name("modulate-uploads", create_if_missing=True)

# Create a volume for songs data
songs_volume = modal.Volume.from_name("modulate-songs", create_if_missing=True)


# Emotion mapping
id2label = {
    "0": "angry", "1": "calm", "2": "disgust", "3": "fearful",
    "4": "happy", "5": "neutral", "6": "sad", "7": "surprised"
}

# Global variables to be set in container
model = None
feature_extractor = None
songs_df = None

def load_models():
    """Load models - called once when container starts"""
    global model, feature_extractor, id2label
    print("Loading models...")
    model_name = "firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53"
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    id2label = {str(int(k)): str(v).strip().lower() for k, v in model.config.id2label.items()}
    print("Models loaded successfully!")

def load_songs_data():
    """Load songs CSV - called once when container starts"""
    global songs_df
    csv_path = Path("/data/songs.csv")
    if csv_path.exists():
        songs_df = pd.read_csv(csv_path)
        print(f"Loaded {len(songs_df)} songs from CSV")
    else:
        print("Warning: songs.csv not found, using empty dataframe")
        songs_df = pd.DataFrame(columns=["artist", "title", "spotify_url", "labels"])

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav'}


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

def trim_silence(audio, sr=16000, threshold=0.01, min_silence_duration=0.1):
    """Remove leading and trailing silence from audio."""
    min_silence_samples = int(min_silence_duration * sr)
    frame_length = 512
    energy = np.array([
        np.sqrt(np.mean(audio[i:i+frame_length]**2))
        for i in range(0, len(audio) - frame_length, frame_length)
    ])
    active_frames = np.where(energy > threshold)[0]
    if len(active_frames) == 0:
        return audio  # all silence, return as-is
    start = max(0, active_frames[0] * frame_length - min_silence_samples)
    end = min(len(audio), (active_frames[-1] + 1) * frame_length + min_silence_samples)
    return audio[start:end]


def predict_mood_result(audio_file_path):
    """
    Predict the emotion from an audio file using chunked inference.
    Splits audio into overlapping 3-second windows (matching model training length),
    averages logits across chunks, then applies temperature scaling for sharper confidence.
    Args:
        audio_file_path: Path to the audio file
    Returns:
        Emotion label as string, confidence score
    """
    CHUNK_SEC = 3.0         # model was trained on ~3s RAVDESS clips
    OVERLAP = 0.5           # 50% overlap between windows
    TEMPERATURE = 9       # <1 sharpens the softmax distribution
    SR = 16000

    try:
        sound_array = load_wav_mono_16k(audio_file_path)

        # Trim leading/trailing silence
        sound_array = trim_silence(sound_array, sr=SR)

        chunk_samples = int(CHUNK_SEC * SR)
        step_samples = int(chunk_samples * (1 - OVERLAP))

        # Build list of chunks; always include at least one
        chunks = []
        if len(sound_array) <= chunk_samples:
            chunks.append(sound_array)
        else:
            for start in range(0, len(sound_array) - chunk_samples + 1, step_samples):
                chunks.append(sound_array[start:start + chunk_samples])
            # Include a final chunk anchored at the end if not already covered
            if (len(sound_array) - chunk_samples) % step_samples != 0:
                chunks.append(sound_array[-chunk_samples:])

        # Run inference on each chunk and accumulate logits
        accumulated_logits = None
        for chunk in chunks:
            inputs = feature_extractor(
                raw_speech=chunk,
                sampling_rate=SR,
                padding=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                logits = model(inputs.input_values.float()).logits
            accumulated_logits = logits if accumulated_logits is None else accumulated_logits + logits

        # Average logits across chunks, then apply temperature scaling
        avg_logits = accumulated_logits / len(chunks)
        scaled_logits = avg_logits / TEMPERATURE

        predicted_id = torch.argmax(scaled_logits, dim=-1).item()
        emotion = id2label[str(predicted_id)]

        probabilities = F.softmax(scaled_logits, dim=-1)
        confidence = probabilities[0, predicted_id].item()

        return emotion, confidence
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")

def get_song_for_mood(mood):
    mood = mood.strip().lower()

    # Map model emotions to uplifting/appropriate CSV labels: calm, fearful, happy, sad
    emotion_map = {
        "angry": "calm",       # calm down angry users
        "disgust": "happy",    # uplift negative feelings
        "neutral": "happy",    # energize neutral mood
        "surprised": "happy",  # match high energy
        "sad": "happy",        # uplift sad users
        "fearful": "calm",      # soothe fearful users
        "calm": "calm",         # match calm mood
        "happy": "happy",       # match happy mood
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
    for i, row in selected.iterrows():
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
            "labels": str(row.get("labels", ""))
        })
    return songs


# Modal web endpoint
@modal_app.function(
    image=image,
    volumes={"/uploads": volume, "/data": songs_volume},
    min_containers=1,  # Keep 1 container warm to reduce cold starts
    timeout=300,  # 5 minute timeout
)
@modal.asgi_app()
def serve():
    """Modal web endpoint that serves the Flask app"""
    # Initialize Flask app
    flask_app = Flask(__name__)
    CORS(flask_app, resources={
        r"/analyze": {"origins": "*", "methods": ["POST", "OPTIONS"]},
        r"/songs": {"origins": "*", "methods": ["POST", "OPTIONS"]},
        r"/health": {"origins": "*", "methods": ["GET", "OPTIONS"]}
    })
    
    flask_app.config['UPLOAD_FOLDER'] = '/uploads'
    flask_app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    
    # Load models and data on container startup
    load_models()
    load_songs_data()
    
    # Register all routes
    @flask_app.route('/')
    def home():
        return "Welcome to Modulate API!"
    
    @flask_app.route('/analyze', methods=['POST', 'OPTIONS'])
    def analyze_audio():
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            file = request.files['audio']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed. Use wav'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            emotion, confidence = predict_mood_result(filepath)
            
            os.remove(filepath)
            
            return jsonify({
                'emotion': emotion,
                'confidence': round(confidence, 4),
                'success': True
            }), 200
        
        except Exception as e:
            return jsonify({'error': str(e), 'success': False}), 500
    
    @flask_app.route('/songs', methods=['POST', 'OPTIONS'])
    def get_songs_route():
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            data = request.get_json()
            if not data or 'emotion' not in data:
                return jsonify({'error': 'No emotion provided'}), 400
            
            emotion = data['emotion'].lower()
            songs = get_song_for_mood(emotion)
            print(songs_df.head())
            return jsonify({
                'songs': songs,
                'success': True
            }), 200
        
        except Exception as e:
            return jsonify({'error': str(e), 'success': False}), 500
    
    @flask_app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'ok'}), 200
    
    return WsgiToAsgi(flask_app)

# Local development server
if __name__ == '__main__':
    # For local development without Modal
    app = Flask(__name__)
    CORS(app, resources={
        r"/analyze": {"origins": "*", "methods": ["POST", "OPTIONS"]},
        r"/songs": {"origins": "*", "methods": ["POST", "OPTIONS"]},
        r"/health": {"origins": "*", "methods": ["GET", "OPTIONS"]}
    })
    
    UPLOAD_FOLDER = 'uploads'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    
    # Load models and data
    load_models()
    load_songs_data()
    
    # Define routes for local development
    @app.route('/')
    def home():
        return "Welcome to Modulate API!"
    
    @app.route('/analyze', methods=['POST', 'OPTIONS'])
    def analyze_audio():
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            file = request.files['audio']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed. Use wav'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            emotion, confidence = predict_mood_result(filepath)
            
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
        print("here")
        if request.method == 'OPTIONS':
            return '', 200
        
        try:
            data = request.get_json()
            if not data or 'emotion' not in data:
                return jsonify({'error': 'No emotion provided'}), 400
            
            emotion = data['emotion'].lower()
            songs = get_song_for_mood(emotion)
            print(songs)
            
            return jsonify({
                'songs': songs,
                'success': True
            }), 200
        
        except Exception as e:
            return jsonify({'error': str(e), 'success': False}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'}), 200
    
    app.run(debug=True, port=5000)

