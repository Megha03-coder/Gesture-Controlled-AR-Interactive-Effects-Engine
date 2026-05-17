import os
import time

# Hide the default Pygame welcome message on startup
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class SoundManager:
    def __init__(self, sound_dir="sounds"):
        # Initialize with more channels for smooth crossfading
        pygame.mixer.init(channels=8)
        self.sounds = {}
        self.sound_dir = sound_dir
        self.current_music = None
        self.bgm_channel = None

        # Pre-load all .wav files as ambient music tracks
        if os.path.exists(self.sound_dir):
            for filename in os.listdir(self.sound_dir):
                if filename.endswith(".wav"):
                    effect_name = filename[:-4]
                    path = os.path.join(self.sound_dir, filename)
                    self.sounds[effect_name] = pygame.mixer.Sound(path)

    def play_effect_music(self, effect_id: str, volume: float = 0.5):
        """Smoothly crossfades to the new effect's music track."""
        if effect_id == self.current_music:
            return  # Already playing this track

        self.current_music = effect_id

        # Fade out current music channel over 1000ms
        if self.bgm_channel and self.bgm_channel.get_busy():
            self.bgm_channel.fadeout(1000)

        if effect_id == "none" or effect_id not in self.sounds:
            return

        # Play new music with 1000ms fade-in
        sound = self.sounds[effect_id]
        sound.set_volume(volume)
        
        # Find an available channel to play the new track
        new_channel = pygame.mixer.find_channel()
        if new_channel:
            new_channel.play(sound, loops=-1, fade_ms=1000)
            self.bgm_channel = new_channel