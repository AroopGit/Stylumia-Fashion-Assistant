import uvicorn
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Get port from environment variable or use default
        port = int(os.getenv("PORT", 8000))
        
        # Start the FastAPI server
        logger.info(f"Starting backend server on port {port}")
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            workers=4
        )
    except Exception as e:
        logger.error(f"Failed to start backend server: {str(e)}")
        raise

if __name__ == "__main__":
    main() 