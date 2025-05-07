# 🗣️🌐 VoiceBridge: AI-Powered Speech-to-Speech Translation for Accessible Education

## Overview
This Flask application transforms YouTube videos through a pipeline that includes:
- Downloading videos using `yt_dlp`
- Extracting audio via `FFmpeg`
- Transcribing audio using `whisper AI`
- Translating transcripts into two languages (one for audio, one for subtitles) using `LangChain` and `ChatGoogleGenerativeAI`
- Generating subtitles in `VTT` format
- Converting translated text to speech using `gTTS`
- Adjusting pitch/tempo for natural sound
- Embedding subtitles and replacing original audio

The final output is a video with new audio and embedded subtitles, accessible via web endpoints for progress and file serving.

## Project Demo:
https://github.com/user-attachments/assets/a5773d99-a6fe-4f0e-b957-f81021c2ee74



## Installation and Setup

### Prerequisites
- Python 3.8+
- `pip` and `virtualenv`
- `FFmpeg` installed
- API keys for `AssemblyAI` and `Google Generative AI`

### Steps
1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/yourprojectname.git
   cd yourprojectname
   ```

2. **Create and activate a virtual environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Install FFmpeg**
   - On Ubuntu:
     ```sh
     sudo apt-get install ffmpeg
     ```
   - On macOS:
     ```sh
     brew install ffmpeg
     ```

5. **Set up API Keys**
   ```sh
   export ASSEMBLYAI_API_KEY='your_assemblyai_api_key'
   export GOOGLE_API_KEY='your_google_api_key'
   ```
   (On Windows, use `set` instead of `export`)

6. **Run the Flask application**
   ```sh
   flask run
   ```
   The web interface will be available at [http://localhost:5000](http://localhost:5000).

## Usage
1. Enter a YouTube video URL.
2. Select target languages for:
   - Re-voicing (Audio)
   - Subtitles
3. Submit the form.
4. Monitor progress via the web interface.
5. Download the processed video with new audio and subtitles.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | POST | Triggers video processing |
| `/progress` | GET | Provides real-time progress updates using SSE |
| `/uploaded_file/<filename>` | GET | Serves processed files (videos, subtitles) |

## Future Improvements
- Sign Language Integration
- Browser Extension
- Improved error handling and logging
- Asynchronous processing with `Celery`
- Enhanced subtitle styling and segmentation
- More natural speech synthesis options

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to your branch (`git push origin feature-name`).
5. Open a pull request.

## Initial Contributers:
1. [Pritam Pani](https://github.com/pritampani)
2. [Saquib Khan](https://github.com/saquib5005)
3. [Swadesmita Sahoo](https://github.com/Swadesmita)
   

## Acknowledgements
- [yt_dlp](https://github.com/yt-dlp/yt-dlp)
- [FFmpeg](https://ffmpeg.org/documentation.html)
- [whisper AI](https://openai.com/index/whisper/)
- [LangChain](https://python.langchain.com/docs/)
- [gTTS](https://pypi.org/project/gTTS/)
- [Flask](https://flask.palletsprojects.com/)

## License
This project is licensed under the MIT License.



## NOTE
The product is hosted, but in render it goes to sleep after some moments. 
If you want to visit and try -> https://googlesolutionchallange.onrender.com


