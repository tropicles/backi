from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Load model and encoder
model_path = Path(__file__).parent / "fraud_detector.pkl"
encoder_path = Path(__file__).parent / "type_encoder.pkl"

# Load the model
model = joblib.load(model_path)

# Load the OneHotEncoder for transaction types
try:
    type_encoder = joblib.load(encoder_path)
except:
    # If encoder not found, we'll handle transaction types differently
    type_encoder = None

@csrf_exempt
def fraud_detector(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            
            if "transactions" not in data:
                return JsonResponse({"error": "Missing 'transactions' field"}, status=400)
            
            transactions = data["transactions"]
            
            if not isinstance(transactions, list):
                return JsonResponse({"error": "'transactions' should be a list of records"}, status=400)
            
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            # Calculate balance difference
            df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
            
            # Process transaction types
            if type_encoder and 'type' in df.columns:
                type_array = df[['type']].values
                type_encoded = type_encoder.transform(type_array)
                
                # Convert to DataFrame
                type_columns = type_encoder.get_feature_names_out(['type'])
                type_df = pd.DataFrame(type_encoded, columns=type_columns, index=df.index)
                
                # Use only the features needed by the model
                X_input = pd.concat([
                    df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                        'oldbalanceDest', 'newbalanceDest', 'balance_diff']],
                    type_df
                ], axis=1)
            else:
                # Fallback if type encoder is not available
                X_input = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                              'oldbalanceDest', 'newbalanceDest', 'balance_diff']]
                              
            # Make predictions
            predictions = model.predict(X_input).tolist()
            
            return JsonResponse({"predictions": predictions}, status=200)
            
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Invalid request method"}, status=400)