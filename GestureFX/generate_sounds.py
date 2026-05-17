import os
import wave
import struct
import math

def generate_ambient_drone(filename, freqs, duration=2.0, volume=0.3):
    sample_rate = 44100
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        omegas = [2.0 * math.pi * f / sample_rate for f in freqs]
        
        for i in range(int(sample_rate * duration)):
            sample = 0
            for w in omegas:
                sample += math.sin(w * i)
            
            sample = sample / len(freqs)
            
            # Add a slow pulse to make it feel musical (completes exactly 1 cycle in 2s)
            pulse = 0.8 + 0.2 * math.sin(2.0 * math.pi * 0.5 * i / sample_rate)
            
            value = int(volume * 32767.0 * sample * pulse)
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)

# Use integer frequencies so they loop perfectly over exactly 2.0 seconds
music_tracks = {
    "fire.wav": [131, 156, 196],           # C3 Minor (Warm)
    "lightning.wav": [330, 392, 466],      # E4 Diminished (Tense)
    "spark.wav": [440, 554, 659],          # A4 Major (Bright)
    "water.wav": [175, 220, 262],          # F3 Major (Flowing)
    "portal.wav": [49, 73, 98],            # G1 Drone (Deep/Dark)
    "smoke.wav": [262, 330, 392, 494],     # C4 Maj7 (Airy)
    "magic_circle.wav": [110, 131, 165],   # A2 Minor (Mystical)
    "success.wav": [262, 330, 392],        # C4 Major (Happy)
    "freeze.wav": [220, 440],              # Octave A (Still)
    "rgb_palm.wav": [196, 247, 294],       # G3 Major (Vibrant)
    "neon_particles.wav": [294, 349, 440], # D4 Minor (Cyberpunk)
    "sepia_filter.wav": [196, 233, 294],   # G3 Minor (Vintage)
}

for name, freqs in music_tracks.items():
    path = os.path.join("sounds", name)
    generate_ambient_drone(path, freqs)
    print(f"Generated ambient music track: {path}")

print("Old sound effects overwritten! Ambient music successfully generated.")