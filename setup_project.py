import os

def create_architecture():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    directories = [
        "tracking",
        "gestures",
        "effects",
        "audio",
        "controls",
        "ui",
        "ai",
        "data",
        "assets/effects",
        "assets/sounds",
        "assets/icons",
        "assets/models",
        "assets/backgrounds"
    ]
    
    print("Initializing VisionFX AI Architecture...")
    for d in directories:
        path = os.path.join(base_dir, d)
        os.makedirs(path, exist_ok=True)
        
        # Create __init__.py for Python packages
        if not d.startswith("assets") and not d.startswith("data"):
            open(os.path.join(path, "__init__.py"), 'a').close()
            
    print("Success! Modular structure created.")

if __name__ == "__main__":
    create_architecture()