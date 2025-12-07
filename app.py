# app.py - Fixed version
from flask import Flask, request, jsonify, render_template, session # Added session
import os
import traceback
import joblib
import pandas as pd # <-- Import pandas
import numpy as np
import google.generativeai as genai

app = Flask(__name__)
# IMPORTANT: Set a secret key for session management
# Replace 'your_very_secret_key' with a real, random secret key in production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "srgec")

# --- Load environment API key and configure Gemini ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("✅ Gemini API configured successfully.")
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")
        GOOGLE_API_KEY = None # Disable Gemini if config fails
else:
    print("⚠️ WARNING: GOOGLE_API_KEY environment variable not set. Chat endpoints will be disabled.")

# --- Load the trained pipeline model ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best_rf_smote_pipeline.pkl") # Make sure this is the correct filename
pipeline = None
try:
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print(f"✅ Loaded pipeline successfully from {MODEL_PATH}")
    else:
        print(f"❌ Error: best_rf_smote_pipeline.pkl not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading pipeline:\n{traceback.format_exc()}")

# --- Define the Feature Names (Must match training order EXACTLY!) ---
expected_feature_names = [
    'Duration_of_alcohol_consumptionyears', 'Total_Bilirubin_mgdl',
    'RBC_million_cellsmicroliter', 'USG_Abdomen_diffuse_liver_or_not',
    'MCHC_gramsdeciliter', 'Direct_mgdl', 'ALPhosphatase_UL',
    'Platelet_Count_lakhsmm', 'Lymphocytes_', 'AG_Ratio', 'SGOTAST_UL',
    'PCV_', 'Total_Count', 'Albumin_gdl', 'Indirect_mgdl'
]

# --- Helper: list available models (debugging) ---
def list_available_models():
    if not GOOGLE_API_KEY:
        return {"error": "API Key not configured"}
    try:
        # Correctly iterate through models if the API returns a list-like object
        models_list = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return models_list
    except Exception as e:
        print(f"Error listing models: {e}")
        return {"error": str(e)}

# --- Home (serves UI) ---
@app.route("/")
def index():
    session.pop('prediction_result', None) # Clear prediction context on reload
    session.pop('chat_history', None) # Clear chat history on reload
    return render_template("index.html")

# --- Diagnosis endpoint (Replaces /predict) ---
# NOTE: Renamed from /predict to /diagnose to match the user's provided code structure
# If your HTML still calls /predict, change this back or update the HTML fetch URL
@app.route("/diagnose", methods=["POST"])
def diagnose():
    """
    Expects JSON: { "features": [ { "feature_name": value, ... } ] }
    OR simplified { "feature_name": value, ... } for single prediction
    Returns JSON: { prediction, probability, explanation }
    """
    if pipeline is None:
        return jsonify({"error": "Prediction pipeline not loaded on server."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON request body."}), 400

    # --- Accommodate both list-of-dicts and single dict input ---
    if isinstance(data, list) and len(data) > 0:
        # Assuming list contains one dictionary for features
        feature_dict = data[0]
    elif isinstance(data, dict):
        # Assuming the dict itself contains the features
        feature_dict = data
    else:
         return jsonify({"error": "Invalid input format. Expected JSON object or list containing one object."}), 400

    # --- FIX 1: Create DataFrame ---
    try:
        # Create DataFrame from the dictionary, ensuring column order
        X_df = pd.DataFrame([feature_dict])
        X_df = X_df[expected_feature_names] # Enforce column order
    except KeyError as e:
         return jsonify({"error": f"Missing feature in input data: {str(e)}"}), 400
    except Exception as e:
        print(f"Error creating DataFrame: {traceback.format_exc()}")
        return jsonify({"error": "Failed to process input features.", "details": str(e)}), 400
    # -------------------------

    try:
        # Use the DataFrame for prediction
        pred = pipeline.predict(X_df)
        pred_label = int(pred[0])

        prob = None
        # Calculate probability correctly
        if hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(X_df)
                # Get probability of the predicted class specifically
                prob = float(probabilities[0, pred_label])
            except Exception as e:
                print(f"Warning: Could not get probability - {e}")
                prob = None # Set to None if predict_proba fails

        # Store prediction in session for chat context
        session['prediction_result'] = pred_label
        session.pop('chat_history', None) # Clear previous chat history

        # --- Prepare the prompt for Gemini explanation ---
        label_text = "possible liver cirrhosis" if pred_label == 1 else "no signs of liver cirrhosis detected"
        confidence_text = f"with {prob*100:.1f}% confidence" if prob is not None else "(confidence unavailable)"

        # Generate a string representation of features for the prompt
        feature_string = "\n".join([f"- {name}: {feature_dict.get(name, 'N/A')}" for name in expected_feature_names])

        prompt = (
            f"You are a helpful AI assistant interpreting results from a health prediction tool. **You are NOT a doctor.** "
            f"A machine learning model predicted **'{label_text}'** {confidence_text} for a user based on the following inputs:\n"
            f"{feature_string}\n\n"
            f"**IMPORTANT:** Do NOT provide medical advice or a diagnosis. Emphasize that this is just a prediction based on patterns in data and they MUST consult a qualified healthcare professional.\n\n"
            f"Please provide the following in simple, clear language for the user:\n"
            f"1. A brief explanation of what this prediction means in general terms (without confirming it as a diagnosis).\n"
            f"2. Mention 3-4 common factors that *can* contribute to liver issues (e.g., alcohol, diet, infections, genetics), without saying these *are* the cause for this user.\n"
            f"3. General next steps: Advise consulting a doctor for proper evaluation, which might include further tests (like blood work, imaging, biopsy if needed).\n"
            f"4. Offer 3-4 general, practical lifestyle suggestions for supporting liver health (e.g., balanced diet, hydration, limiting alcohol/toxins, exercise) - applicable regardless of the prediction.\n"
            f"5. Include a clear disclaimer stating this is not medical advice and a doctor consultation is essential.\n"
            f"Keep the entire response concise (around 150-200 words)."
        )


        explanation = "(AI explanation unavailable.)" # Default
        if GOOGLE_API_KEY:
            try:
                model_name = "models/gemini-flash-latest" # Or "gemini-pro"
                model = genai.GenerativeModel(model_name)

                # --- FIX 2a: Correct generate_content call ---
                response = model.generate_content(prompt)
                # ------------------------------------------

                # --- FIX 2b: Simplify response extraction ---
                explanation = response.text
                # -----------------------------------------

            except Exception as e:
                error_detail = str(e)
                explanation = f"(Failed to generate AI explanation: {error_detail})"
                print(f"Gemini Error in /diagnose: {traceback.format_exc()}")
                # Try listing models only if the error seems related to model name
                if "not found" in error_detail or "supported" in error_detail:
                    debug_models = list_available_models()
                    explanation += f" | Debug Info (Available Models/Error): {debug_models}"
        else:
            explanation = "(AI explanation unavailable because GOOGLE_API_KEY is not configured on the server.)"

        return jsonify({
            "prediction": pred_label,
            "probability": prob, # Send probability of the predicted class
            "explanation": explanation
        })

    except Exception as e:
        print(f"Diagnose error (Prediction/Processing): {traceback.format_exc()}")
        return jsonify({"error": "Diagnosis failed during prediction or processing.", "details": str(e)}), 500


# --- Follow-up chat endpoint ---
# --- Follow-up chat endpoint ---
@app.route("/chat", methods=["POST"])
def chat():
    """
    Expects: { "message": "your question" }
    Uses Gemini to answer follow-up questions, using context from session.
    """
    if not GOOGLE_API_KEY:
        return jsonify({"error": "Chatbot is not configured (API key missing)."}), 500

    data = request.get_json(silent=True)
    if not data or not data.get("message") or not isinstance(data["message"], str):
        return jsonify({"error": "Invalid or missing 'message' in request body."}), 400

    user_message = data["message"].strip()
    if not user_message:
         return jsonify({"error": "Message cannot be empty."}), 400

    # Get prediction context from session (used for logic, not sent directly as system role)
    prediction_result = session.get('prediction_result', None)
    # (Context string logic can remain if you want to use it for other purposes, but we won't send it as 'system')

    # --- Build the Message History for Gemini ---
    if 'chat_history' not in session:
        session['chat_history'] = []

    # Prepare message history for the API - ONLY user and model roles
    messages_for_api = [] # Start with an empty list

    # Add stored history (if any) - Limit history length
    history = session.get('chat_history', [])
    MAX_HISTORY = 10 # Keep last 10 turns (user + model)
    # --- FIX: Only add valid 'user' and 'model' roles from history ---
    messages_for_api.extend(history[-MAX_HISTORY:])
    # -----------------------------------------------------------------

    # Add the current user message
    current_user_message_dict = {"role": "user", "parts": [user_message]}
    messages_for_api.append(current_user_message_dict)

    # --- Add System Instructions (if supported by the model/library in this way) ---
    # Define the system prompt instructions separately
    system_instruction_text = f"""
    You are a helpful AI assistant for a liver health prediction tool. **You MUST NOT provide medical advice, diagnosis, or treatment.**
    Your primary role is to offer *general* information related to liver health, lifestyle, and potentially clarifying the *type* of information a doctor might discuss.
    **NEVER** interpret specific user symptoms or test results.
    **ALWAYS** strongly recommend consulting a qualified healthcare professional for any personal health concerns or decisions.
    Keep responses empathetic, concise, and easy to understand. Use bullet points for lists.
    """
    # -------------------------------------------------------------------------------


    try:
        model_name = "models/gemini-flash-latest" # Or "gemini-pro-latest" / "models/gemini-2.5-flash"
        # --- FIX: Pass system instruction separately if possible ---
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction_text # Pass instructions here
        )
        # ---------------------------------------------------------

        # --- FIX: Call generate_content with ONLY user/model history ---
        response = model.generate_content(messages_for_api) # Pass the list containing only user/model turns
        # ------------------------------------------------------------

        bot_response = response.text

        # Update chat history in session (only user and model messages)
        # Don't save the system instruction in the turn-by-turn history
        session['chat_history'].append(current_user_message_dict) # Add the user message we just sent
        session['chat_history'].append({"role": "model", "parts": [bot_response]}) # Add the bot response
        # Ensure history doesn't grow indefinitely
        if len(session['chat_history']) > MAX_HISTORY + 2:
             session['chat_history'] = session['chat_history'][-(MAX_HISTORY + 2):] # Prune older messages
        session.modified = True

        return jsonify({"reply": bot_response})

    except Exception as e:
        error_detail = str(e)
        print(f"Chat error: {traceback.format_exc()}")
        debug_models_info = list_available_models()
        return jsonify({
            "error": "Chat failed to get response.",
            "details": error_detail,
            "available_models_or_error": debug_models_info
        }), 500

# --- (Rest of your app.py code remains the same) ---

if __name__ == "__main__":
    # ... (startup code remains the same) ...
    app.run(debug=True, host="0.0.0.0", port=5000)