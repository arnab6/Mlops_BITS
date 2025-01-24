import requests
import numpy as np

# Define the API endpoint
url = "http://127.0.0.1:9696/predict"

# Example feature data 
example_features = [-0.8840273411734237,
  1.7822804597528,
  -1.6934743932256737,
  1.5495725621015788,
  -0.9932045660129188,
  -0.7254956645460121,
  -2.479298392851497,
  1.209813994449899,
  -2.2661700485522998,
  -5.28117971890821,
  2.0735085990671527,
  -3.12282392388905,
  0.6607144843420577,
  -4.917824090499457,
  0.6257200002150141,
  -4.333929180751305,
  -6.577996207565781,
  -3.536506148902853,
  2.0769497730863744,
  0.7274025170236523,
  0.8975468709797324,
  0.09308641009425725,
  -0.3671041959569382,
  -0.9026815800521959,
  0.40058091666261214,
  1.4719238541413673,
  1.6353838846258901,
  0.7677944749032589,
  -1.075859376790042]
# Prepare the payload
payload = {
    "features": example_features
}

# Send POST request to the API
response = requests.post(url, json=payload)

# Display the response
if response.status_code == 200:
    print("Prediction Response:", response.json())
else:
    print("Error:", response.status_code, response.text)