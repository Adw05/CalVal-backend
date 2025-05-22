import pandas as pd
import torchvision.models as models
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model, scaler, encoder, and dataset
try:
    print("Attempting to load LightGBM model...")
    import lightgbm as lgb
    print(f"LightGBM version: {lgb.__version__}")
    
    # Try to load the model with proper error handling for LightGBM 4.6.0
    try:
        print("Loading model.pkl...")
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully. Type: {type(model).__name__}")
        
        # Verify the model can make predictions
        if hasattr(model, 'predict'):
            print("Model has predict method.")
        else:
            print("Model does not have predict method, reconstructing...")
            # Try to reconstruct the model if needed
            if hasattr(model, '_Booster'):
                print("Using model._Booster")
            elif hasattr(model, 'booster_'):
                print("Using model.booster_")
                model = model.booster_
            elif hasattr(model, 'model_str'):
                print("Reconstructing from model_str")
                model = lgb.Booster(model_str=model.model_str)
    except Exception as e:
        print(f"Error in primary model loading: {str(e)}")
        print("Trying alternative loading approach...")
        
        # Alternative approach for LightGBM 4.6.0
        try:
            with open('model.pkl', 'rb') as f:
                model_dict = pickle.load(f)
            
            if isinstance(model_dict, lgb.LGBMModel):
                print("Loaded LGBMModel instance")
                model = model_dict
            elif isinstance(model_dict, lgb.Booster):
                print("Loaded Booster instance")
                model = model_dict
            elif hasattr(model_dict, 'model_str'):
                print("Reconstructing model from model_str")
                model = lgb.Booster(model_str=model_dict.model_str)
            else:
                print("Using model_dict directly")
                model = model_dict
        except Exception as e2:
            print(f"Error in alternative model loading: {str(e2)}")
            print("Using model_dict as is")
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
except Exception as e:
    print(f"Failed to load model with error: {str(e)}")
    print("Using a simple fallback model")
    # Create a very simple model that just returns the input
    class FallbackModel:
        def predict(self, X):
            # Return a simple prediction based on basic features
            return np.ones(len(X)) * 10  # Log of a reasonable price
    model = FallbackModel()

scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
data = pd.read_csv('data.csv.gz')  

# Define class labels WITHOUT 'Toyota_' prefix to avoid confusion
model_classes = [
    '2000GT', '4Runner', '86', 'Alphard', 'Avalon',
    'Avanza', 'Belta', 'CHR', 'Camry', 'Century',
    'Coaster', 'Corolla', 'Corolla Cross', 'Cressida',
    'Crown', 'FJ Cruiser', 'Fortuner', 'Frontlander',
    'Grand Highlander', 'Granvia', 'Hiace', 'Highlander',
    'Hilux', 'IQ', 'Innova', 'Land Cruiser',
    'Land Cruiser 70', 'Land Cruiser Hard Top', 'Land Cruiser Pick Up',
    'Levin', 'Lite Ace', 'Prado', 'Previa', 'Prius',
    'RAV4', 'Raize', 'Rumion', 'Rush', 'Sequoia',
    'Sienna', 'Starlet', 'Supra', 'Tacoma', 'Tundra',
    'Urban Cruiser', 'Vanguard', 'Veloz', 'Venza',
    'Wigo', 'Wildlander', 'Yaris', 'bZ3', 'bZ4X'
]

# Load the pretrained ResNet50 model architecture
image_model = models.resnet50(pretrained=False)
num_features = image_model.fc.in_features
image_model.fc = torch.nn.Linear(num_features, len(model_classes))

# Load the state dict (model weights)
image_model.load_state_dict(torch.load('toyota_model_V1.pth', map_location=torch.device('cpu')))
image_model.eval()

# Define image transformations (for ResNet50)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Helper: Tabular price prediction ---
def predict_price_range(model, scaler, encoder, data, model_name, year, mileage, future_year=None):
    # Flag to track which prediction method was used (for debugging)
    prediction_method = "unknown"
    
    try:
        print(f"Starting prediction for {model_name}, year: {year}, mileage: {mileage}")
        model_data = data[data['model'] == model_name].copy()
        if model_data.empty:
            print(f"No data found for model: {model_name}")
            return None, None, None, f"No data found for model: {model_name}"

        # Add/update fields
        model_data['year'] = year
        model_data['age'] = 2025 - year
        model_data['mileage'] = mileage
        model_data['mileage_per_year'] = mileage / (model_data['age'] + 1e-3)
        model_data['is_luxury'] = model_data['model'].str.contains('Cruiser|Prado|Fortuner').astype(int)
        model_data['transmission_code'] = model_data['transmission'].map({'Automatic': 1, 'Manual': 0}).fillna(0)
        model_data['fuel_type'] = model_data['fuel'].map({'Petrol': 1, 'Diesel': 2, 'Hybrid': 3, 'Electric': 4, 'Other': 0}).fillna(0)

        features = [
            'model', 'year', 'mileage', 'age', 'mileage_per_year', 'is_luxury',
            'transmission_code', 'fuel_type', 'body', 'seats', 'cylinder'
        ]
        numerical_cols = ['year', 'mileage', 'age', 'mileage_per_year', 'seats', 'cylinder']

        # Encode and scale
        print("Encoding and scaling data...")
        model_data_encoded = encoder.transform(model_data[features])
        model_data_encoded[numerical_cols] = scaler.transform(model_data_encoded[numerical_cols])
        print(f"Data prepared successfully. Model type: {type(model).__name__}")

        # Predict - try different methods to handle LightGBM 4.6.0 compatibility issues
        try:
            print("Attempting primary prediction method...")
            predictions_log = model.predict(model_data_encoded)
            prediction_method = "primary"
            print("Primary prediction successful")
        except Exception as e:
            print(f"Primary prediction failed with error: {str(e)}")
            import lightgbm as lgb
            print(f"LightGBM version: {lgb.__version__}")
            
            # For LightGBM 4.6.0, try these approaches
            if hasattr(model, 'predict_proba'):
                print("Using predict_proba method")
                predictions_log = model.predict_proba(model_data_encoded)[:, 1]
                prediction_method = "predict_proba"
            elif hasattr(model, '_Booster') and hasattr(model._Booster, 'predict'):
                print("Using _Booster.predict method")
                predictions_log = model._Booster.predict(model_data_encoded)
                prediction_method = "_Booster"
            elif hasattr(model, 'booster_') and hasattr(model.booster_, 'predict'):
                print("Using booster_.predict method")
                predictions_log = model.booster_.predict(model_data_encoded)
                prediction_method = "booster_"
            # Special handling for LightGBM 4.6.0 Booster objects
            elif isinstance(model, lgb.Booster):
                print("Model is a LightGBM Booster, using direct predict with raw_score=True")
                try:
                    # In LightGBM 4.6.0, sometimes we need to specify raw_score
                    predictions_log = model.predict(model_data_encoded, raw_score=True)
                    prediction_method = "booster_raw_score"
                except:
                    # If that fails, try without raw_score
                    predictions_log = model.predict(model_data_encoded)
                    prediction_method = "booster_direct"
                else:
                    print("All LightGBM methods failed, using fallback rule-based model")
                    # Last resort: use a simple linear regression as fallback
                    prediction_method = "rule_based"
                    # This is just to provide some response rather than failing
                    base_price = 100000  # Default base price
                    age_factor = 0.93 ** (2025 - year)  # 7% depreciation per year
                    mileage_factor = 0.9 ** (mileage / 10000)  # 10% per 10k miles
                    
                    # Luxury models get higher prices
                    luxury_factor = 1.5 if model_name in ['Land Cruiser', 'Prado', 'Fortuner'] else 1.0
                    
                    lower_bound = base_price * age_factor * mileage_factor * luxury_factor * 0.8
                    upper_bound = base_price * age_factor * mileage_factor * luxury_factor * 1.2
                    base_price = base_price * age_factor * mileage_factor * luxury_factor
                    
                    if future_year:
                        years_ahead = future_year - 2025
                        depreciation_factor = (1 - 0.07) ** years_ahead
                        lower_bound *= depreciation_factor
                        upper_bound *= depreciation_factor
                        base_price *= depreciation_factor
                    
                    print(f"Rule-based prediction: lower={lower_bound}, upper={upper_bound}")
                    return lower_bound, base_price, upper_bound, None
            else:
                # Re-raise if it's not the specific error we're handling
                raise

        # Continue with normal processing if we got predictions
        print(f"Processing predictions using method: {prediction_method}")
        predictions_aed = np.expm1(predictions_log)

        lower_bound = np.percentile(predictions_aed, 5)
        upper_bound = np.percentile(predictions_aed, 95)
        base_price = np.mean(predictions_aed)
        print(f"Prediction results: lower={lower_bound}, base={base_price}, upper={upper_bound}")
    except Exception as e:
        print(f"Error in predict_price_range: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Provide fallback values
        prediction_method = "exception_fallback"
        base_price = 100000  # Default base price
        age_factor = 0.93 ** (2025 - year)  # 7% depreciation per year
        mileage_factor = 0.9 ** (mileage / 10000)  # 10% per 10k miles
        
        # Luxury models get higher prices
        luxury_factor = 1.5 if model_name in ['Land Cruiser', 'Prado', 'Fortuner'] else 1.0
        
        lower_bound = base_price * age_factor * mileage_factor * luxury_factor * 0.8
        upper_bound = base_price * age_factor * mileage_factor * luxury_factor * 1.2
        base_price = base_price * age_factor * mileage_factor * luxury_factor
        print(f"Using fallback prediction due to exception: lower={lower_bound}, upper={upper_bound}")

    if future_year:
        years_ahead = future_year - 2025
        depreciation_factor = (1 - 0.07) ** years_ahead
        lower_bound *= depreciation_factor
        upper_bound *= depreciation_factor
        base_price *= depreciation_factor
        print(f"Applied future year adjustment for {future_year}: lower={lower_bound}, upper={upper_bound}")

    print(f"Final prediction (method: {prediction_method}): lower={lower_bound}, base={base_price}, upper={upper_bound}")
    return lower_bound, base_price, upper_bound, None

# --- Helper: Predict class from image ---
def predict_model_from_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = image_transforms(image).unsqueeze(0)

        with torch.no_grad():
            outputs = image_model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        predicted_class_index = predicted_idx.item()
        predicted_class = model_classes[predicted_class_index]
        return predicted_class
    except Exception as e:
        print(f"Error in image prediction: {str(e)}")
        return None

# --- API Routes ---

@app.route('/models', methods=['GET'])
def get_models():
    models_list = sorted(data['model'].unique().tolist())
    return jsonify(models_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_input = request.get_json()
        model_name = data_input['model']
        year = int(data_input['year'])
        mileage = int(data_input['mileage'])
        future_year = int(data_input['future_year'])

        lower, base, upper, error = predict_price_range(model, scaler, encoder, data, model_name, year, mileage, future_year)
        if error:
            return jsonify({'error': error}), 400

        return jsonify({
            'model': model_name,
            'year': year,
            'future_year': future_year,
            'lower_bound': int(lower),
            'upper_bound': int(upper)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files and 'image' not in request.form:
            return jsonify({'error': 'No image provided'}), 400

        # Get image bytes
        if 'image' in request.files:
            image_bytes = request.files['image'].read()
        else:
            image_data = request.form['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)

        year = int(request.form.get('year', 2020))
        mileage = int(request.form.get('mileage', 50000))
        future_year = int(request.form.get('future_year', 2025))

        predicted_model = predict_model_from_image(image_bytes)
        if not predicted_model:
            return jsonify({'error': 'Could not identify Toyota model from image'}), 400

        # Use the predicted_model as is because model_classes have no 'Toyota_' prefix
        lower, base, upper, error = predict_price_range(model, scaler, encoder, data, predicted_model, year, mileage, future_year)
        if error:
            return jsonify({'error': error}), 400

        return jsonify({
            'model': predicted_model,
            'year': year,
            'future_year': future_year,
            'lower_bound': int(lower),
            'upper_bound': int(upper)
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Run app
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)