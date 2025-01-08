# 🎙️ Voice Typing with Whisper AI  

Effortlessly convert your speech into text using OpenAI's **Whisper** speech-to-text model! This project demonstrates two innovative approaches to voice typing:  

1. **Keypress-Based Voice Typing**: Press and hold a key to record your voice, release it to process, and the text appears wherever your cursor is placed.  
2. **Automatic Voice Detection**: Using Silero VAD (Voice Activity Detection), the system detects when you’re speaking and transcribes it automatically.  

---

## 📽️ Demo Video  

Watch the full demo on YouTube to see the project in action:  

[![Voice Typing Demo](https://img.youtube.com/vi/8npo8QoIIgw/maxresdefault.jpg)](https://youtu.be/8npo8QoIIgw)  

---

## 🚀 Features  
- **Hands-Free Typing**: Type without touching the keyboard.  
- **Customizable Hotkey**: Choose your preferred key for recording.  
- **Automatic Voice Detection**: No need to press a key; just speak, and the system handles the rest.  
- **Whisper Model Integration**: Leverages OpenAI's Whisper for accurate transcription.  

---

## 🛠️ How It Works  
- **Keypress-Based Method**:  
  1. Press and hold a specific key to start recording.  
  2. Release the key to process and type the transcribed text.  

- **VAD-Based Method**:  
  1. The system uses Silero VAD to detect when you're speaking.  
  2. Transcribes the audio and types it wherever the cursor is.  

---

## 📂 Code and Setup  
1. Clone this repository:  
   ```bash
   git clone https://github.com/biplob004/voice-typing-whisper
   cd voice-typing-whisper
   ```

2. Install python packages:
   ```bash
      pip3 install -r requirements.txt
   ```
3. Run the python file:
   ```bash
   python3 main_manual.py
   # or...
   python3 main_automatic.py
   ```
