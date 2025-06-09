# Deployment Guide

This guide will help you deploy the Stylumia Fashion Assistant without Docker.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Step 1: Clone the Repository

```bash
git clone https://github.com/AroopGit/Stylumia-Fashion-Assistant.git
cd Stylumia-Fashion-Assistant
```

## Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Prepare Data

Make sure you have the following data files in the `data/processed` directory:
- `processed_metadata.csv`
- `image_features.json`

## Step 5: Deploy Backend

1. Open a new terminal window
2. Activate the virtual environment
3. Run the backend server:
```bash
python scripts/deploy_backend.py
```

The backend will be available at `http://localhost:8000`

## Step 6: Deploy Frontend

1. Open another terminal window
2. Activate the virtual environment
3. Run the frontend server:
```bash
python scripts/deploy_frontend.py
```

The frontend will be available at `http://localhost:8501`

## Environment Variables

You can customize the deployment using these environment variables:

- `PORT`: Backend server port (default: 8000)
- `STREAMLIT_PORT`: Frontend server port (default: 8501)

## Troubleshooting

1. If you get a "Module not found" error:
   - Make sure you're in the virtual environment
   - Try reinstalling dependencies: `pip install -r requirements.txt`

2. If the backend fails to start:
   - Check if port 8000 is available
   - Verify that data files exist in the correct location

3. If the frontend fails to start:
   - Check if port 8501 is available
   - Verify that the backend is running and accessible

## Production Deployment

For production deployment, consider:

1. Using a process manager (e.g., PM2, Supervisor)
2. Setting up a reverse proxy (e.g., Nginx)
3. Implementing proper security measures
4. Setting up monitoring and logging
5. Using environment variables for sensitive data

## Support

For any deployment issues, please open an issue on GitHub. 