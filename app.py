from flask import Flask, request
import torch
from model import CpGPredictor

# Initialize Flask app
app = Flask(__name__)

# Load the entire model from the .pth file
model = torch.load('cpg_detector_model.pth', map_location=torch.device('cpu'))
print(type(model))

# Set the model to evaluation mode
model.eval()


# Set the model to evaluation mode
model.eval()


# Define function to preprocess input and make prediction
def make_prediction(dna_sequence):
    # Alphabet helpers
    alphabet = 'NACGT'
    dna2int = {a: i for a, i in zip(alphabet, range(5))}

    # Preprocess input sequence (convert to list of integers)
    int_sequence = [dna2int[char] for char in dna_sequence]
    print("Preprocessed input sequence:", int_sequence)

    # Convert to PyTorch tensor and add batch dimension
    tensor_sequence = torch.tensor(int_sequence).unsqueeze(0)
    print("Input tensor shape:", tensor_sequence.shape)

    # Make prediction using the loaded model
    with torch.no_grad():
        prediction = model(tensor_sequence)

    print("Predicted output: ", prediction.item())

    return prediction.item()


# Define route for home page
@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
    <title>CpG Detector</title>
    </head>
    <body>
    <h1>CpG Detector</h1>
    <form method="post" action="/predict">
        <label for="dna_sequence">Enter DNA sequence:</label><br>
        <input type="text" id="dna_sequence" name="dna_sequence" value=""><br><br>
        <input type="submit" value="Predict">
    </form>
    </body>
    </html>
    """


# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input sequence from form
    input_sequence = request.form.get("dna_sequence")

    # Make prediction
    if input_sequence:
        prediction = make_prediction(input_sequence)
        return f'Predicted CpG count: {prediction}'
    else:
        return 'Please enter a DNA sequence.'


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
