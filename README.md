# Interactive AI Assistant - Render Deployment Guide

This repository contains an interactive AI assistant that combines educational AI capabilities with enhanced URL analysis. The application automatically detects whether you're asking a question or analyzing a URL, and routes your query to the appropriate component.

## Features

- Educational AI assistant using OpenAI's API
- Enhanced URL analysis with improved accuracy
- Automatic query classification
- Comprehensive security assessment
- User-friendly interface

## Deployment on Render

### Prerequisites

- A [Render](https://render.com/) account (free tier available)
- A GitHub account to host your repository
- An OpenAI API key for the educational AI assistant functionality

### Step 1: Prepare Your Repository

1. Create a new GitHub repository
2. Upload all the files from this package to your repository:
   - `app.py` - The main Streamlit application
   - `assistant_components.py` - The core components of the AI assistant
   - `requirements.txt` - Dependencies required by the application
   - `runtime.txt` - Python version specification
   - `Procfile` - Instructions for Render on how to run the application

### Step 2: Deploy on Render

1. Log in to your Render account
2. Click on the "New +" button and select "Web Service"
3. Connect your GitHub account if you haven't already
4. Select the repository containing your AI assistant
5. Configure the deployment with the following settings:
   - **Name**: Choose a name for your service (e.g., "ai-assistant")
   - **Environment**: Python
   - **Region**: Choose the region closest to your users
   - **Branch**: main (or your default branch)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
6. Click "Create Web Service"

Render will automatically detect the Python version from your `runtime.txt` file and install the dependencies from `requirements.txt`. The deployment process may take a few minutes.

### Step 3: Access Your Application

Once the deployment is complete, Render will provide you with a URL to access your application (e.g., `https://ai-assistant.onrender.com`). You can share this URL with others to let them use your AI assistant.

## Using the Application

1. Open the application URL in your web browser
2. Enter your OpenAI API key in the sidebar
3. Ask educational questions or enter URLs to analyze
4. The application will automatically detect the type of query and provide an appropriate response

## Important Notes

- The free tier on Render has some limitations:
  - Your application will "sleep" after 15 minutes of inactivity
  - When a sleeping application receives a new request, it may take up to 30 seconds to "wake up"
  - There are monthly usage limits (check Render's documentation for details)
- Your OpenAI API key is required for the educational AI assistant functionality
- The application does not store your API key permanently; you'll need to enter it each time you use the application

## Updating Your Application

To update your application:

1. Make changes to your local files
2. Commit and push the changes to your GitHub repository:
   ```
   git add .
   git commit -m "Description of changes"
   git push
   ```
3. Render will automatically detect the changes and redeploy your application

## Troubleshooting

If you encounter issues with your deployment:

1. Check the Render logs for error messages
2. Verify that all required files are present in your repository
3. Ensure your `requirements.txt` file includes all necessary dependencies
4. Check that your OpenAI API key is valid and has sufficient quota

For more detailed information, refer to the [Render documentation](https://render.com/docs) or contact Render support.
