import pandas as pd
import requests
import json
import subprocess
import sys
import time
import cleantxty
from typing import List, Dict, Union

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
            self.process = subprocess.Popen(
                [self.ollama_path, "serve"],
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
    def __init__(self, model_name: str = "deepseek-r1:14b", feedback_file: str = "labeled_data.csv"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.feedback_file = feedback_file
        self.issue_categories = [
            "Other",
            "App Issues",
            "Gift Card Not Available",
            "24 Hour Blocked ( Fraud Prevention )",
            "Cash Removal From Wallet / Never Received",
            "No Reward/ Why No Reward ( If he is eligible )",
            "Blocked Account / Under Review",
            "Can't Answer Survey ( Survey Limit )"
        ]
        self.labeled_data = self.load_feedback()

    def load_feedback(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.feedback_file)
        except FileNotFoundError:
            return pd.DataFrame(columns=["ComplaintMessage", "Issue Classification"])

    def save_feedback(self):
        self.labeled_data.to_csv(self.feedback_file, index=False)

    def check_historical_data(self, complaint: str) -> Union[str, None]:
        match = self.labeled_data[self.labeled_data["ComplaintMessage"] == complaint]
        if not match.empty:
            return match["Issue Classification"].iloc[0]
        return None

    def generate_prompt(self, complaint: str, keywords: List[str]) -> str:
        categories_list = "\n".join([f"- {category}" for category in self.issue_categories])
        return f"""Given the following customer complaint and extracted keywords, classify it into ONE category from the list below:

Complaint: {complaint}
Keywords: {', '.join(keywords)}

Categories:
{categories_list}

Respond with ONLY ONE category name. No explanation needed."""

    def classify_issue(self, complaint: str, keywords: List[str]) -> str:
        # Check for existing classification in labeled data
        historical_classification = self.check_historical_data(complaint)
        if historical_classification:
            return historical_classification

        # If no historical match, use the model
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

    def update_feedback(self, complaint: str, classification: str):
        new_row = {"ComplaintMessage": complaint, "Issue Classification": classification}
        self.labeled_data = self.labeled_data.append(new_row, ignore_index=True)
        self.save_feedback()


def extract_keywords(text: str) -> List[str]:
    cleaned_text = cleantxty.clean(
        text,
        default_case="lower",
        remove_punctuations=True,
        remove_digits=True
    )
    words = cleaned_text.split()
    issue_keywords = ["redeem", "voucher", "coupon", "coins", "provide", "showing", "gift", "card", "reward", "blocked", "fraud"]
    return [word for word in words if word in issue_keywords]


# Load the dataset
file_path = "customer_support_data.csv"
df = pd.read_csv(file_path)

# Initialize Ollama Server
server = OllamaServer()
if not server.start_server():
    sys.exit("Failed to start Ollama server. Exiting...")

# Initialize classifier
classifier = IssueClassifier("deepseek-r1:14b")


def process_complaints(row):
    complaint = row.get("ComplaintMessage", "")
    keywords = extract_keywords(complaint)
    issue_type = classifier.classify_issue(complaint, keywords)
    classifier.update_feedback(complaint, issue_type)  # Save feedback
    return pd.Series([issue_type, keywords])


# Process the complaints
df[["Issue Classification", "Keywords"]] = df.apply(process_complaints, axis=1)

# Save the classified data
output_file = "classified_customer_support.csv"
df.to_csv(output_file, index=False)

server.stop_server()

print(f"Classification and keyword extraction completed and saved to {output_file}")