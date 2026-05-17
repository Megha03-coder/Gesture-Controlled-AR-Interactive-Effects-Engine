class Config:
    # Project Metadata
    PROJECT_NAME = "VisionFX AI"
    VERSION = "1.0.0"
    
    # Camera & Vision Settings
    CAMERA_INDEX = 0
    RESOLUTION = (1280, 720)
    TARGET_FPS = 60
    
    # AI Confidence Thresholds
    HAND_DETECTION_CONFIDENCE = 0.7
    FACE_DETECTION_CONFIDENCE = 0.7
    POSE_DETECTION_CONFIDENCE = 0.7
    
    # HUD & UI
    HUD_COLOR_MAIN = (0, 255, 255)  # Cyan (Iron Man / Sci-Fi style)
    HUD_COLOR_ALERT = (0, 0, 255)   # Red
    
    # Feature Toggles
    ENABLE_VOICE_ASSISTANT = True
    ENABLE_SYSTEM_CONTROL = True    # Set to True to allow PC control (mouse/volume)