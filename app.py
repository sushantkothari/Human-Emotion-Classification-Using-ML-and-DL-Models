from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Input
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import h5py
import json

app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

print("Loading model and preprocessing files...")

def decode_if_bytes(value):
    """Helper function to decode bytes to string if needed"""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value

def inspect_h5_file(filepath):
    """Inspect the contents of the H5 file"""
    try:
        with h5py.File(filepath, 'r') as f:
            # Try to get model configuration
            if 'model_config' in f.attrs:
                config_str = decode_if_bytes(f.attrs['model_config'])
                try:
                    config = json.loads(config_str)
                    print("Model configuration found:")
                    return config
                except json.JSONDecodeError as e:
                    print(f"Error parsing model configuration: {e}")
            
            # If no config found, print available keys
            print("Available keys in H5 file:", list(f.keys()))
            if 'model_weights' in f:
                print("Model weights found. Layer names:")
                for layer_name in f['model_weights'].keys():
                    print(f"- {layer_name}")
            
            return None
    except Exception as e:
        print(f"Error inspecting H5 file: {e}")
        return None

def create_compatible_model(vocab_size, max_len):
    """Create a model with compatible architecture"""
    model = Sequential()
    
    # Add layers with minimal configuration to avoid version conflicts
    model.add(Input(shape=(max_len,)))
    model.add(Embedding(vocab_size, 100))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
    
    return model

# Load label encoder and vocabulary info
try:
    with open('lb1.pkl', 'rb') as f:
        lb = pickle.load(f)
    with open('vocab_info.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    
    vocab_size = vocab_info['vocab_size']
    max_len = vocab_info['max_len']
    print(f"Loaded vocab_size: {vocab_size}, max_len: {max_len}")
    print(f"Available emotion labels: {lb.classes_}")
except Exception as e:
    print(f"Error loading supporting files: {e}")
    raise

# Create and load the model
try:
    print("Creating compatible model...")
    model = create_compatible_model(vocab_size, max_len)
    
    print("Initializing model...")
    dummy_input = np.zeros((1, max_len))
    model(dummy_input)
    
    print("Loading weights...")
    try:
        # Try direct loading first
        model.load_weights('model1.h5', by_name=True)
    except Exception as e1:
        print(f"Direct weight loading failed: {e1}")
        try:
            # Try loading layer by layer
            with h5py.File('model1.h5', 'r') as f:
                if 'model_weights' in f:
                    for layer in model.layers:
                        if layer.name in f['model_weights']:
                            weight_group = f['model_weights'][layer.name]
                            if 'kernel:0' in weight_group:
                                weights = []
                                for weight_name in ['kernel:0', 'bias:0']:
                                    if weight_name in weight_group:
                                        weight_value = weight_group[weight_name][:]
                                        weights.append(weight_value)
                                if weights:
                                    layer.set_weights(weights)
                                    print(f"Loaded weights for layer: {layer.name}")
        except Exception as e2:
            print(f"Layer-by-layer loading failed: {e2}")
            raise
    
    # Compile the model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    print("Model loaded successfully")
    model.summary()
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def sentence_cleaning(sentence):
    """Clean and preprocess input text"""
    stemmer = PorterStemmer()
    print(f"Original text: {sentence}")
    
    text = re.sub("[^a-zA-Z]", " ", sentence)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    
    print(f"Cleaned text: {text}")
    
    one_hot_word = [one_hot(input_text=text, n=vocab_size)]
    pad = pad_sequences(sequences=one_hot_word, maxlen=max_len, padding='pre')
    
    print(f"Padded shape: {pad.shape}")
    return pad

def predict_emotion(input_text):
    """Predict emotion from input text"""
    try:
        cleaned_text = sentence_cleaning(input_text)
        prediction = model.predict(cleaned_text, verbose=0)
        
        predicted_class_index = np.argmax(prediction, axis=-1)[0]
        probability = prediction[0][predicted_class_index]
        emotion = lb.inverse_transform([predicted_class_index])[0]
        
        print(f"Predicted emotion: {emotion}")
        print(f"Confidence: {probability:.2%}")
        
        return emotion, probability
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text'].strip()
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': 'No text provided'
            })
        
        emotion, probability = predict_emotion(text)
        
        return jsonify({
            'status': 'success',
            'emotion': emotion,
            'probability': f"{probability:.2%}"
        })
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)