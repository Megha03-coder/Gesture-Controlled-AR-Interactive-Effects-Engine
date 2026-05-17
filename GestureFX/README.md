# GestureFX AI — Hand Gesture Controlled Visual Effects (Real-time)

A real-time computer-vision application that uses your webcam + MediaPipe Hands to detect hand landmarks, recognize finger gestures, and trigger Snapchat/Instagram-style AR visual effects and sounds.

## Features
- Real-time webcam processing (OpenCV)
- Hand landmark tracking (MediaPipe)
- Gesture detection via finger-count patterns
- Gesture-triggered effects:
  - Fire aura (finger-following particle glow)
  - Lightning (electric lines + sparks)
  - Sparkle system (alpha-fading particles)
  - RGB filter (dynamic channel shifting)
  - Neon glow (blur-based glow layers)
  - Background blur (mask + selective Gaussian blur)
  - Particle engine (spawn/move/fade)
  - Freeze frame / slow-motion (when closed fist)
- Sound playback (Pygame)
- Modular, scalable architecture

> This project uses code-generated visuals (no required image assets). An `assets/` folder is included for optional future PNG/SFX drops.

## Installation
### 1) Create/activate a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Gestures (finger patterns)
Gestures are recognized using finger-up patterns (thumb handling is simplified; keep your hand fairly centered and facing the camera).

| Gesture | Finger pattern | Effect |
|---|---|---|
| One Finger Up | `[0,1,0,0,0]` | Fire aura |
| Two Fingers Up | `[0,1,1,0,0]` | Lightning |
| Open Palm | `[1,1,1,1,1]` | RGB filter + Rainbow overlay + background blur |
| Closed Fist | `[0,0,0,0,0]` | Freeze frame + slow motion |
| Thumbs Up | `[1,0,0,0,0]` | Success animation + sound |
| Victory Sign | `[0,1,1,0,0]` | Neon glow + animated particles |

## Project Structure
```text
GestureFX/
  main.py
  hand_tracking.py
  gesture_detector.py
  effect_manager.py
  particle_system.py
  overlay_manager.py
  sound_manager.py
  utils.py
  assets/
    effects/
    sounds/
    videos/
  requirements.txt
  README.md
```

