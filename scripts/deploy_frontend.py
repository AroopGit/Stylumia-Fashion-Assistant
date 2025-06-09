import streamlit.web.cli as stcli
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Get port from environment variable or use default
        port = int(os.getenv("STREAMLIT_PORT", 8501))
        
        # Set Streamlit configuration
        os.environ["STREAMLIT_SERVER_PORT"] = str(port)
        os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
        
        # Start the Streamlit server
        logger.info(f"Starting frontend server on port {port}")
        sys.argv = ["streamlit", "run", "streamlit_app/app.py"]
        sys.exit(stcli.main())
    except Exception as e:
        logger.error(f"Failed to start frontend server: {str(e)}")
        raise

if __name__ == "__main__":
    main() 