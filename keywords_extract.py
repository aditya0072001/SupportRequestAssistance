import pandas as pd
import requests
from typing import List

class DeepSeekClassifier:
    def __init__(self, api_url: str):
        self.api_url = api_url
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

    def generate_prompt(self, complaint: str) -> str:
        categories_list = "\n".join([f"- {category}" for category in self.issue_categories])
        return f"""Given the following customer complaint, classify it into ONE category from the list below:

        Complaint: {complaint}

        Categories:
        {categories_list}

        Respond with ONLY ONE category name. No explanation needed."""

    def classify_issue(self, complaint: str) -> str:
        if not isinstance(complaint, str) or complaint.strip() == "":
            return "Other"

        try:
            prompt = self.generate_prompt(complaint)
            payload = {"prompt": prompt, "temperature": 0.3}
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()

            # Extract classification from response
            result = response.json().get("outputs", [{}])[0].get("text", "").strip()
            return result if result else "Other"
        except Exception as e:
            print(f"Error classifying complaint: {e}")
            return "Other"

# ✅ Load the dataset
file_path = "customer_support_data.csv"
df = pd.read_csv(file_path)

# ✅ Initialize the classifier
api_url = "http://127.0.0.1:8266/v1/completions"  # Replace with vLLM API endpoint
classifier = DeepSeekClassifier(api_url)

def process_complaints(row):
    complaint = row.get("ComplaintMessage", "")
    issue_type = classifier.classify_issue(complaint)
    return pd.Series([issue_type])

# ✅ Process the complaints
df["Issue Classification"] = df.apply(process_complaints, axis=1)

# ✅ Save the classified data
output_file = "classified_customer_support.csv"
df.to_csv(output_file, index=False)

print(f"Classification completed and saved to {output_file}")