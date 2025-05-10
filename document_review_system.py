from typing import List, Dict, Any
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import docx
import os
import json
import sys
from dotenv import load_dotenv

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

# 5. LLM rule processor using LangChain (Gemini)
def apply_llm_rule(rule: Rule, text: str, llm) -> Dict[str, Any]:
    prompt = f"{rule.llm_prompt}\n\nDocument Content:\n{text}"
    response = llm([HumanMessage(content=prompt)])
    return {
        "rule_id": rule.rule_id,
        "category": rule.category,
        "status": "review",
        "message": response.content,
        "suggestion": "Review LLM feedback."
    }

# 6. Main review engine
def review_document_text(text: str, rules: List[Rule]) -> List[Dict[str, Any]]:
    results = []
    
    # Fetch LLM configurations from environment variables
    llm_model = os.getenv("LLM_MODEL", "gemini-pro")
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", 0.2))
    
    # Fetch the API key from environment variable
    genie_api_key = os.getenv("GENIE_API_KEY")
    
    if not genie_api_key:
        raise ValueError("GENIE_API_KEY is missing in the .env file")

    # Initialize LangChain LLM using environment settings and API key
    llm = ChatGoogleGenerativeAI(
        model=llm_model, 
        temperature=llm_temperature, 
        api_key=genie_api_key  # Passing the API key here
    )

    for rule in rules:
        if rule.type == "static_length_check":
            result = apply_static_rule(rule, text)
        elif rule.type == "llm_check":
            result = apply_llm_rule(rule, text, llm)
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
