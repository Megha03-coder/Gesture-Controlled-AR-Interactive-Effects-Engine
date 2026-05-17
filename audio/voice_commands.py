import speech_recognition as sr
import pyttsx3
import threading

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
        # Test initialization of speech engine
        self._test_engine = pyttsx3.init()
        self._configure_voice(self._test_engine)
        
        self.speak("Vision F X systems online and listening.")

    def _configure_voice(self, engine):
        """Configures the pyttsx3 engine to use a suitable voice and rate."""
        voices = engine.getProperty('voices')
        # Try to find a female/AI-sounding voice (often Zira on Windows)
        for voice in voices:
            if "zira" in voice.name.lower() or "female" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 170)

    def speak(self, text: str):
        """Speaks text in a separate thread. Creating a new engine instance per thread 
        is safer on Windows to avoid COM object thread-affinity errors."""
        def _speak_thread():
            local_engine = pyttsx3.init()
            self._configure_voice(local_engine)
            local_engine.say(text)
            local_engine.runAndWait()
            
        threading.Thread(target=_speak_thread, daemon=True).start()

    def listen(self) -> str:
        """Listens for audio from the default microphone and transcribes it."""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
            try:
                audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"[AI Scanner] Voice Command: '{command}'")
                return command
            except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
                return ""