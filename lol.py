import pandas as pd
import requests
import json
import subprocess
import sys
import time
import os
from typing import List, Dict, Tuple, Optional

class OllamaServer:
    def __init__(self):
        """Initialize Ollama server manager"""
        self.ollama_path = self._get_ollama_path()
        self.process = None
        
    def _get_ollama_path(self) -> str:
        """
        Get the path to Ollama executable based on OS
        Returns:
            str: Path to Ollama executable
        """
        if sys.platform == "win32":
            return "ollama.exe"
        elif sys.platform == "darwin":
            return "/usr/local/bin/ollama"
        else:  # Linux
            return "/usr/bin/ollama"
    
    def is_running(self) -> bool:
        """
        Check if Ollama server is running
        Returns:
            bool: True if server is running, False otherwise
        """
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def start_server(self) -> bool:
        """
        Start the Ollama server
        Returns:
            bool: True if server started successfully, False otherwise
        """
        if self.is_running():
            print("Ollama server is already running")
            return True
            
        try:
            print("Starting Ollama server...")
            if sys.platform == "win32":
                self.process = subprocess.Popen(
                    [self.ollama_path, "serve"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                self.process = subprocess.Popen(
                    [self.ollama_path, "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Wait for server to start
            max_retries = 10
            for i in range(max_retries):
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
        """Stop the Ollama server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Ollama server stopped")

class ModelManager:
    def __init__(self, model_name: str = "deepseek-r1:14b"):
        """
        Initialize model manager
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        
    def is_model_downloaded(self) -> bool:
        """
        Check if the model is already downloaded
        Returns:
            bool: True if model exists, False otherwise
        """
        try:
            response = requests.get("http://localhost:11434/api/tags")
            models = response.json().get("models", [])
            return any(model["name"] == self.model_name for model in models)
        except:
            return False
    
    def download_model(self) -> bool:
        """
        Download the specified model
        Returns:
            bool: True if download successful, False otherwise
        """
        if self.is_model_downloaded():
            print(f"Model {self.model_name} is already downloaded")
            return True
            
        try:
            print(f"Downloading model {self.model_name}...")
            response = requests.post(
                "http://localhost:11434/api/pull",
                json={"name": self.model_name}
            )
            
            if response.status_code == 200:
                print(f"Model {self.model_name} downloaded successfully")
                return True
            else:
                print(f"Failed to download model: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False

class IssueClassifier:
    def __init__(self, model_name: str = "llama2"):
        """
        Initialize the issue classifier with Ollama model
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.issue_types = [
            "Age", "App Related", "DataSecurity", "Gender",
            "Giftcard", "Location", "Other", "Reward", "Survey"
        ]
        
    def generate_prompt(self, complaint: str) -> str:
        """
        Generate a prompt for the LLM to classify the issue
        Args:
            complaint (str): The complaint message to classify
        Returns:
            str: Formatted prompt for the LLM
        """
        return f"""Given the following customer complaint, classify it into ONE of these categories:
Categories: {', '.join(self.issue_types)}

Complaint: {complaint}

Respond with ONLY ONE category name from the list above. No explanation needed.
"""

    def classify_issue(self, complaint: str) -> str:
        """
        Classify a single complaint using Ollama
        Args:
            complaint (str): The complaint message to classify
        Returns:
            str: Classified issue type
        """
        if not complaint or pd.isna(complaint):
            return "Other"
            
        try:
            payload = {
                "model": self.model_name,
                "prompt": self.generate_prompt(complaint),
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            # Extract the response text
            result = response.json()["response"].strip()
            
            # Validate that the response is one of our issue types
            if result in self.issue_types:
                return result
            else:
                return "Other"
                
        except Exception as e:
            print(f"Error classifying complaint: {e}")
            return "Other"

def process_csv(input_file: str, output_file: str, model_name: str = "deepseek-r1:14b", batch_size: int = 50) -> bool:
    """
    Process the CSV file and classify all complaints
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        model_name (str): Name of the Ollama model to use
        batch_size (int): Number of complaints to process before saving
    Returns:
        bool: True if processing successful, False otherwise
    """
    try:
        # Initialize Ollama server
        server = OllamaServer()
        if not server.start_server():
            return False
            
        # Initialize and download model
        model_manager = ModelManager(model_name)
        if not model_manager.download_model():
            return False
            
        # Initialize classifier
        classifier = IssueClassifier(model_name)
        
        # Read CSV file
        df = pd.read_csv(input_file)
        
        # Create a copy of the dataframe to store results
        results_df = df.copy()
        
        # Initialize progress counter
        total_complaints = len(df)
        processed = 0
        
        print(f"Starting classification of {total_complaints} complaints...")
        
        # Process complaints in batches
        for index, row in df.iterrows():
            if pd.notna(row['ComplaintMessage']):
                issue_type = classifier.classify_issue(str(row['ComplaintMessage']))
                results_df.at[index, 'Issue Type'] = issue_type
            else:
                results_df.at[index, 'Issue Type'] = 'Other'
                
            processed += 1
            
            # Save progress after each batch
            if processed % batch_size == 0:
                results_df.to_csv(output_file, index=False)
                print(f"Processed {processed}/{total_complaints} complaints...")
                time.sleep(1)  # Small delay to prevent overwhelming the API
        
        # Final save
        results_df.to_csv(output_file, index=False)
        print(f"Classification completed! Results saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return False
    finally:
        if 'server' in locals():
            server.stop_server()

def main():
    """Main function to run the classification process"""
    INPUT_FILE = "Customer Support 13f30c9f1fc780f4b67fc9f4c5c06359.csv"
    OUTPUT_FILE = "classified_customer_support.csv"
    MODEL_NAME = "deepseek-r1:14b"  # or any other model you have in Ollama
    
    if process_csv(INPUT_FILE, OUTPUT_FILE, MODEL_NAME):
        print("Processing completed successfully")
    else:
        print("Processing failed")

if __name__ == "__main__":
    main()