from google.cloud import aiplatform
from google.api_core import exceptions

# Initialize the AI platform client
aiplatform.init(project="mineral-oxide-421911", location="us-central1")

try:
    # List available models in the Vertex AI endpoint
    models = aiplatform.Model.list()

    for model in models:
        print(f"Available model: {model.display_name}")
except exceptions.GoogleAPICallError as e:
    print(f"Error listing models: {e}")
