# CpG Predictor Model Inferencing using PyTorch and LSTM from Local System

## Contents

1. [Clone Model to Local System](#clone-model-to-local-system)
2. [Flask App Execution to Inference the CpG Predictor Model](#inference-cpg-predictor)
3. [Steps to Test with Sample Results](#execution-sample-results)

## Clone Model to Local System
Feel free to clone the model using either of the below links:
- In case of HTTPS, use ```git clone https://github.com/kraviteja95/CpGDetector.git```
- In case of SSH, use ```git clone git@github.com:kraviteja95/CpGDetector.git```

## Flask App Execution to Inference the CpG Predictor Model
- Navigate to the CpGDetector folder where you have cloned and run the Flask application by opening the terminal in that path. Look at the below example for reference.
  - ```bash
    cd ~/flask_poc/CpGDetector
    ```
  - ```bash
    python3 app.py
    ```
- You will notice a URL in the terminal when you execute ```python3 app.py```. Open that. Note that the URL would be - http://127.0.0.1:5000. This is nothing but your localhost http://localhost:5000.
- A webpage opens like below.

  <img width="215" alt="image" src="https://github.com/kraviteja95/CpGDetector/assets/74393760/1e5f93cf-8aa6-4096-9be9-3561009b99f6">

- Provide the DNA sequence as the input and click on ```Predict``` button. You will notice the predicted output from the window.
  
  <img width="386" alt="image" src="https://github.com/kraviteja95/CpGDetector/assets/74393760/3e756aca-7de5-4dd5-a2ac-4910d88d6092">


  
