import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import os
import speech_recognition as sr
import re
import subprocess
import nltk
import gtts
import simpleaudio as sa


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# File paths for saving the model and vectorizer
MODEL_FILE = 'prompt_classification_model.pkl'
VECTORIZER_FILE = 'prompt_classification_vectorizer.pkl'

# Ensure the directories for audio files and text records exist
AUDIO_DIR = "audio_files"
LOCAL_DATA = "text_records"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(LOCAL_DATA, exist_ok=True)

# Speech recognition and playback utility functions
output_id = 0

def play_audio(filename):
    """Plays the specified wav file using simpleaudio."""
    if os.path.isfile(filename):
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait until sound has finished playing
    else:
        print(f"Audio file {filename} does not exist.")

def convert_mp3_to_wav(mp3_filename, wav_filename):
    """Converts mp3 to wav since simpleaudio handles wav files."""
    try:
        subprocess.run(['ffmpeg', '-y', '-i', mp3_filename, wav_filename], check=True)
    except subprocess.CalledProcessError:
        print(f"Error converting {mp3_filename} to {wav_filename}. Ensure FFmpeg is installed.")

def speak_save(text, lang='en'):
    """Speaks and saves the provided text to an audio file."""
    global output_id
    mp3_filename = os.path.join(AUDIO_DIR, f"output_{output_id}.mp3")
    wav_filename = os.path.join(AUDIO_DIR, f"output_{output_id}.wav")
    
    # Generate TTS audio file
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(mp3_filename)
    
    # Convert mp3 to wav
    convert_mp3_to_wav(mp3_filename, wav_filename)
    
    # Play the audio
    play_audio(wav_filename)
    
    output_id += 1

def clean_text(text):
    """Cleans the input text for further processing."""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Check for commands, for email automation and weather API
def contains_email(text):
    """Check if the input text contains the keyword 'email'."""
    return 'email' in text.lower() or 'emails' in text.lower() or 'emailing' in text.lower()

# Load the dataset
train_df = pd.read_csv('file1.csv', engine='python', on_bad_lines='skip')

# Clean the text data
train_df['statement'] = train_df['statement'].apply(clean_text)

# Define the feature and target variables
X = train_df['statement']
y = train_df['type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    print("Training the model...")

    # Create a TF-IDF vectorizer with tuned parameters
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Try Logistic Regression as an alternative model
    clf = LogisticRegression(max_iter=1000)  # Increased iterations for convergence

    # Hyperparameter tuning for Logistic Regression
    params = {'C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(clf, params, cv=5)
    grid_search.fit(X_train_vectorized, y_train)

    clf = grid_search.best_estimator_

    # Save the model and vectorizer
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print("Model and vectorizer saved.")
else:
    print("Loading the saved model and vectorizer...")

    # Load the saved model and vectorizer
    clf = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

# Transform the test data using the loaded vectorizer
X_test_vectorized = vectorizer.transform(X_test)

# Evaluate the classifier on the testing data
y_pred = clf.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Speech recognition for new text input
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    speak_save("Please speak something...")
    recognizer.adjust_for_ambient_noise(source)
    

    try:
        audio = recognizer.listen(source, timeout=None)
        new_text = recognizer.recognize_google(audio)
        speak_save(f"You said: {new_text}")

        # Classify the recognized text
        new_text_cleaned = clean_text(new_text)
        new_text_vectorized = vectorizer.transform([new_text_cleaned])
        prediction = clf.predict(new_text_vectorized)
        result_message = f"Classification result: {prediction[0]}"
        print(result_message)

        if prediction[0] == 'question' and contains_email(new_text_cleaned):
            # Launch automate_email.py 
            print("Launching email function...")
            subprocess.run(['python3', 'automate_email.py'])

    except sr.UnknownValueError:
        speak_save("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        speak_save(f"Could not request results from Google Speech Recognition service; {e}")
