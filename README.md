## SpeechRecognition_and_Text2Speech_App
An automatic speech recognition and text-to-speech app using open-source models from Hugging Face. Supports speech recognition from both microphone audio and recorded audio.

## Demo Video

Watch the demo video to see how to run the app:

[![Watch the video](https://img.youtube.com/vi/0LIqNsrkmx0/maxresdefault.jpg)](https://www.youtube.com/watch?v=0LIqNsrkmx0)


## Prerequisites

To run the app locally, you need to have `ffmpeg` and `espeak` installed on your system. You can install them using the following commands:

### For ffmpeg

- **On Ubuntu:**
  ```bash
  sudo apt update
  sudo apt install ffmpeg


- **On macOS:**
  ```bash
  brew install ffmpeg

## Downloading FFmpeg for Windows

To download FFmpeg for Windows, please follow these steps:

1. Go to the official [FFmpeg download page](https://ffmpeg.org/download.html).

2. Select the appropriate version for Windows and download the zip file.

3. Extract the zip file to a directory of your choice.

For detailed instructions and further information, please refer to the [official FFmpeg documentation](https://ffmpeg.org/documentation.html).

## Downloading eSpeak

### macOS

1. Install eSpeak using Homebrew:
   ```sh
   brew install espeak


### Ubuntu

To install eSpeak using `apt-get`, run the following commands in your terminal:

       
        sudo apt-get update
        sudo apt-get install espeak


### Windows

To install eSpeak on Windows, follow these steps:

1. Download the eSpeak installer from the [official eSpeak download page](http://espeak.sourceforge.net/download.html).
2. Run the installer.
3. Follow the on-screen instructions to complete the installation.

## Run Locally

To run the app locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/mdmahmudun/SpeechRecognition_and_Text2Speech_App.git

2. Navigate to the project directory:

    ```bash
    cd SpeechRecognition_and_Text2Speech_App

3. Install required dependencies
    ```bash
    pip install -r requirements.txt

4. Run the app:
    ```bash
    python app.py

