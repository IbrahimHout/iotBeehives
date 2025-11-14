# Beehive Monitoring System

This project is a smart beehive monitoring system that uses a Raspberry Pi to continuously listen to a beehive, classify its status using an AI model, and provide real-time data visualization and alerts.

## Features

*   **Real-Time Audio Classification:** Continuously monitors beehive sounds and classifies the hive's status (e.g., "QueenBee Present" or "QueenBee Absent").
*   **AI-Powered:** Uses a convolutional neural network (CNN) to analyze audio data and make predictions.
*   **Data Visualization:** Integrates with InfluxDB and Grafana to provide a real-time dashboard of the beehive's status.
*   **SMS Alerts:** Automatically sends an SMS alert to the farmer when a potentially dangerous state (like an absent queen) is detected.
*   **Raspberry Pi Based:** Designed to run on a low-cost, low-power Raspberry Pi 3 B+.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Set Up the Python Environment

We recommend using a Python virtual environment to manage the project's dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install System Dependencies

This project requires PortAudio for audio recording. On Debian-based systems like Raspberry Pi OS, you can install it with:

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev
```

### 4. Train the AI Model (on a Laptop/Desktop)

The model training process is computationally intensive and should be done on a more powerful machine, not the Raspberry Pi.

**a. Download the Dataset:**

*   Go to the Kaggle dataset page: [BeeHive Audio Dataset with Queen and without Queen](https://www.kaggle.com/datasets/harshkumar1711/beehive-audio-dataset-with-queen-and-without-queen)
*   Download the dataset and unzip it.
*   Copy the `Dataset` directory into the `data` directory of this project.

**b. Run the Training Script:**

```bash
python src/train_model.py
```

This will train the model and save it as `models/beehive_model.h5`.

### 5. Run the Real-Time Classifier (on the Raspberry Pi)

**a. Transfer the Project:**

*   Copy the entire project directory (including the trained `beehive_model.h5` file) to your Raspberry Pi.

**b. Configure the Classifier:**

*   Open `src/real_time_classifier.py` and update the following configurations:
    *   **InfluxDB:** Set `INFLUXDB_HOST`, `INFLUXDB_PORT`, and `INFLUXDB_DATABASE`.
    *   **SMS Alerts:** Set `FREESMS_USER` and `FREESMS_PASS` with your Free Mobile credentials.

**c. Run the Script:**

```bash
python src/real_time_classifier.py
```

The script will start listening to the beehive and sending data to your InfluxDB database.

## Running with Docker (Recommended)

This project is fully containerized, which makes it easy to deploy and manage.

### 1. Install Docker and Docker Compose

*   Follow the official instructions to install Docker and Docker Compose on your Raspberry Pi:
    *   [Install Docker Engine on Raspberry Pi OS](https://docs.docker.com/engine/install/raspberry-pi-os/)
    *   [Install Docker Compose](https://docs.docker.com/compose/install/)

### 2. Build and Run the System

```bash
docker-compose up --build
```

This command will:
*   Build the Docker image for the beehive application.
*   Start the beehive app, InfluxDB, and Grafana services.
*   The beehive app will start monitoring the beehive, and you can access the Grafana dashboard at `http://<your-raspberry-pi-ip>:3000`.
