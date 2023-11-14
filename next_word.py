from flask import Flask, render_template, request, jsonify
import joblib  # For loading the .pkl model

app = Flask(__name__)

# Load the trained model
model = joblib.load("tokenizer.pkl")

@app.route('/next_word')
def next_word():
    return render_template('option_three.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Assuming your model expects a certain format for input data
        input_data = data['input_data']

        # Make predictions using the loaded model
        predictions = model.predict([input_data])

        # Return the predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e})

if __name__ == '__main__':
    app.run(debug=True)
