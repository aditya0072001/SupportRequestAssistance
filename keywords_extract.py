import pandas as pd
import requests
import json
import subprocess
import sys
import time
import os
import cleantxty
from typing import List, Dict, Tuple, Optional

class OllamaServer:
    def __init__(self):
        self.ollama_path = self._get_ollama_path()
        self.process = None
        
    def _get_ollama_path(self) -> str:
        if sys.platform == "win32":
            return "ollama.exe"
        elif sys.platform == "darwin":
            return "/usr/local/bin/ollama"
        else:
            return "/usr/bin/ollama"
    
    def is_running(self) -> bool:
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def start_server(self) -> bool:
        if self.is_running():
            print("Ollama server is already running")
            return True
        
        try:
            print("Starting Ollama server...")
            if sys.platform == "win32":
                self.process = subprocess.Popen([
                    self.ollama_path, "serve"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                self.process = subprocess.Popen([
                    self.ollama_path, "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            max_retries = 10
            for _ in range(max_retries):
                if self.is_running():
                    print("Ollama server started successfully")
                    return True
                time.sleep(1)
            
            print("Failed to start Ollama server")
            return False
            
        except Exception as e:
            print(f"Error starting Ollama server: {e}")
            return False
    
    def stop_server(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Ollama server stopped")

class IssueClassifier:
    def __init__(self, model_name: str = "deepseek-r1:14b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
    
    def generate_prompt(self, complaint: str, keywords: List[str]) -> str:
        return f"""Given the following customer complaint and extracted keywords, classify it into ONE category:

Complaint: {complaint}
Keywords: {', '.join(keywords)}

Respond with ONLY ONE category name. No explanation needed."""
    
    def classify_issue(self, complaint: str, keywords: List[str]) -> str:
        if not complaint or pd.isna(complaint):
            return "Other"
            
        try:
            payload = {
                "model": self.model_name,
                "prompt": self.generate_prompt(complaint, keywords),
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            return response.json()["response"].strip()
                
        except Exception as e:
            print(f"Error classifying complaint: {e}")
            return "Other"

def extract_keywords(text: str) -> List[str]:
    cleaned_text = cleantxty.clean(
        text, 
        default_case="lower", 
        remove_punctuations=True, 
        remove_digits=True
    )
    words = cleaned_text.split()
    issue_keywords = ["redeem", "voucher", "coupon", "coins", "provide", "showing"]
    extracted_keywords = [word for word in words if word in issue_keywords]
    return extracted_keywords

# Load the dataset
file_path = "customer_support_data.csv"
df = pd.read_csv(file_path)

# Initialize Ollama Server
server = OllamaServer()
server.start_server()

# Initialize classifier
classifier = IssueClassifier("deepseek-r1:14b")

def process_complaints(row):
    keywords = extract_keywords(row["ComplaintMessage"])
    issue_type = classifier.classify_issue(row["ComplaintMessage"], keywords)
    return pd.Series([issue_type, keywords])

df[["Issue Type", "Keywords"]] = df.apply(process_complaints, axis=1)

df.to_csv("classified_customer_support.csv", index=False)

server.stop_server()

print("Classification and keyword extraction completed and saved to classified_customer_support.csv")
