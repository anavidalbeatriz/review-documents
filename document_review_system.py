from typing import List, Dict, Any
from pydantic import BaseModel
import docx
import os
import json
import sys
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.api_core import exceptions

# Load environment variables from .env
load_dotenv()

# 1. Rule schema
class Rule(BaseModel):
    rule_id: str
    description: str
    type: str  # 'static_length_check', 'llm_check', etc.
    category: str  # e.g., 'clarity', 'cohesion', etc.
    min_words: int = 0
    llm_prompt: str = ""


# 2. Load rules from JSON file
def load_rules(json_path: str) -> List[Rule]:
    with open(json_path, 'r', encoding='utf-8') as f:
        rule_dicts = json.load(f)
    return [Rule(**r) for r in rule_dicts]

# 3. Document parser (whole document)
def extract_full_text(doc_path: str) -> str:
    doc = docx.Document(doc_path)
    full_text = " ".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
    return full_text

# 4. Static rule processor
def apply_static_rule(rule: Rule, text: str) -> Dict[str, Any]:
    word_count = len(text.split())
    passed = word_count >= rule.min_words
    return {
        "rule_id": rule.rule_id,
        "category": rule.category,
        "status": "pass" if passed else "fail",
        "message": f"{word_count} words found. Minimum required is {rule.min_words}.",
        "suggestion": "Expand the document." if not passed else ""
    }

# 5. LLM rule processor using Vertex AI (Generative AI Model)
def apply_llm_rule(rule: Rule, text: str) -> Dict[str, Any]:
    # Initialize Vertex AI client
    try:
        # Get environment variables for Vertex AI setup
        project_id = os.getenv("GCP_PROJECT_ID")
        region = os.getenv("GCP_REGION", "us-central1")
        vertex_model_id = os.getenv("VERTEX_AI_MODEL_ID")  # The generative model's ID
        
        # Initialize the AI platform client
        aiplatform.init(project=project_id, location=region)

        # Prepare the prompt for the generative model
        prompt = f"{rule.llm_prompt}\n\nDocument Content:\n{text}"

        # Call Vertex AI's generative model
        model = aiplatform.Model(vertex_model_id)
        response = model.predict([prompt])

        return {
            "rule_id": rule.rule_id,
            "category": rule.category,
            "status": "review",
            "message": response.predictions[0],  # Assuming the first prediction is the model's response
            "suggestion": "Review the LLM feedback."
        }
    except exceptions.GoogleAPICallError as e:
        return {
            "rule_id": rule.rule_id,
            "category": rule.category,
            "status": "error",
            "message": f"Error with Vertex AI: {e}",
            "suggestion": "Check your API settings or model availability."
        }

# 6. Main review engine
def review_document_text(text: str, rules: List[Rule]) -> List[Dict[str, Any]]:
    results = []
    
    for rule in rules:
        if rule.type == "static_length_check":
            result = apply_static_rule(rule, text)
        elif rule.type == "llm_check":
            result = apply_llm_rule(rule, text)
        else:
            result = {
                "rule_id": rule.rule_id,
                "category": rule.category,
                "status": "error",
                "message": f"Unknown rule type: {rule.type}"
            }

        results.append(result)

    return results


# 7. Sample usage
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        rule_path = sys.argv[1]
        input_text = " ".join(sys.argv[2:])
        
        # Fetch the rules JSON path from environment variable or default to provided path
        rules_json_path = os.getenv("RULES_JSON_PATH", rule_path)
        
        rules = load_rules(rules_json_path)
        feedback = review_document_text(input_text, rules)
        
        for item in feedback:
            print("\nRule:", item["rule_id"])
            print("Category:", item["category"])
            print("Status:", item["status"])
            print("Message:", item["message"])
            if "suggestion" in item:
                print("Suggestion:", item["suggestion"])
    else:
        print("Usage: python document_review_system.py <rules.json> <text_to_analyze>")
