import pyaudio
import numpy as np
import tensorflow as tf
import librosa
import time
from influxdb import InfluxDBClient
import freesms

# --- Constants ---
MODEL_PATH = 'models/beehive_model.h5'
SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 128
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
FRAMES_PER_BUFFER = CHUNK * (SAMPLE_RATE // CHUNK) # ~1 second of audio

# --- InfluxDB Configuration ---
INFLUXDB_HOST = 'localhost' # Change to your InfluxDB host
INFLUXDB_PORT = 8086
INFLUXDB_DATABASE = 'beehive' # Change to your database name

# --- SMS Alert Configuration ---
FREESMS_USER = 'your_freesms_user' # Change to your Free Mobile user ID
FREESMS_PASS = 'your_freesms_password' # Change to your Free Mobile password
ALERT_COOLDOWN = 3600 # 1 hour in seconds
last_alert_time = 0

# --- Load the Trained Model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have trained the model and it is located at:", MODEL_PATH)
    exit()

# --- SMS Alert Function ---
def send_sms_alert(message):
    """
    Sends an SMS alert if the cooldown period has passed.
    """
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time > ALERT_COOLDOWN:
        try:
            client = freesms.Client(FREESMS_USER, FREESMS_PASS)
            client.send(message)
            last_alert_time = current_time
            print("SMS alert sent successfully.")
        except Exception as e:
            print(f"Error sending SMS alert: {e}")

# --- Audio Recording and Classification ---
def classify_audio_stream():
    """
    Records audio from the microphone and classifies it in real-time.
    """
    p = pyaudio.PyAudio()
    
    # --- InfluxDB Connection ---
    try:
        client = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT)
        client.switch_database(INFLUXDB_DATABASE)
        print("Connected to InfluxDB.")
    except Exception as e:
        print(f"Error connecting to InfluxDB: {e}")
        return

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER)
    
    print("Listening for beehive sounds...")
    
    try:
        while True:
            frames = []
            for _ in range(0, int(SAMPLE_RATE / FRAMES_PER_BUFFER * DURATION)):
                data = stream.read(FRAMES_PER_BUFFER)
                frames.append(data)
            
            # Convert audio frames to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0

            # Preprocess the audio and predict
            if len(audio_data) > 0:
                mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE, n_mels=N_MELS)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Reshape for the model
                mel_spec_db = mel_spec_db[np.newaxis, ..., np.newaxis]

                # Predict
                prediction = model.predict(mel_spec_db)
                confidence = prediction[0][0]
                status = "QueenBee Present" if confidence > 0.5 else "QueenBee Absent"
                
                print(f"Status: {status} (Confidence: {confidence:.2f})")
                
                # --- Send SMS Alert if Queen is Absent ---
                if status == "QueenBee Absent":
                    send_sms_alert("Alert: Queen bee may be absent from hive_1.")
                
                # --- Write to InfluxDB ---
                json_body = [
                    {
                        "measurement": "beehive_status",
                        "tags": {
                            "hive_id": "hive_1"
                        },
                        "fields": {
                            "status": status,
                            "confidence": float(confidence)
                        }
                    }
                ]
                client.write_points(json_body)

    except KeyboardInterrupt:
        print("Stopping the classifier.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    classify_audio_stream()
