"""
Setup script for celeste-ai-gym
"""
import subprocess
import sys
import os

def install_requirements():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ Dependencies installed successfully")

def create_directories():
    """Create necessary directories"""
    directories = ["models", "logs", "checkpoints"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directories created")

def main():
    print("Setting up Celeste AI Gym...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    try:
        install_requirements()
        create_directories()
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Start Celeste with the AI mod enabled")
        print("2. Run: python main.py")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()