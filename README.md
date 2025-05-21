# Toyota Vehicle API Backend

This is a Flask-based API for Toyota vehicle price prediction and model identification.

## Deployment to Render

### Option 1: Deploy via Render Dashboard

1. Create a new account or log in to [Render](https://render.com/)
2. Click on "New +" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - Name: toyota-vehicle-api (or your preferred name)
   - Environment: Docker
   - Branch: main (or your default branch)
   - Root Directory: /backend (if your repo includes other directories)
   - Leave other settings as default
5. Click "Create Web Service"

### Option 2: Deploy via render.yaml

1. Push your code to a GitHub repository
2. Log in to [Render](https://render.com/)
3. Click on "New +" and select "Blueprint"
4. Connect your GitHub repository
5. Render will automatically detect the render.yaml file and set up the service

## Local Development

To run the application locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

To build and run with Docker:

```bash
# Build the Docker image
docker build -t toyota-vehicle-api .

# Run the container
docker run -p 8080:8080 toyota-vehicle-api
```

## API Endpoints

- `GET /models`: Get all available Toyota models
- `POST /predict`: Predict price range for a specific model
- `POST /predict_image`: Identify Toyota model from image and predict price range
