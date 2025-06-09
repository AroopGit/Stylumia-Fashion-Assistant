import os
import subprocess
from pathlib import Path

def main():
    # Change to the frontend directory
    frontend_dir = Path(__file__).parent.parent / "frontend"
    os.chdir(frontend_dir)
    
    # Run npm start
    subprocess.run(["npm", "start"])

if __name__ == "__main__":
    main() 