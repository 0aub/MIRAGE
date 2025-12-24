
import requests
import os
import sys
import glob
from pathlib import Path

# API Endpoint (Double files prefix due to router inclusion)
API_URL = "http://localhost:8000/files/files/upload"

def ingest_all():
    # Path to docs inside Docker container
    docs_dir = Path("/app/docs_to_ingest")
    
    print(f"Looking for files in: {docs_dir}")
    
    files = list(docs_dir.glob("*.[pP][dD][fF]")) + list(docs_dir.glob("*.[tT][xX][tT]"))
    
    if not files:
        print("No files found!")
        return

    print(f"Found {len(files)} files.")
    
    for file_path in files:
        print(f"Uploading {file_path.name}...", end=" ")
        
        try:
            with open(file_path, "rb") as f:
                files_payload = {"file": (file_path.name, f, "application/pdf" if file_path.suffix.lower() == ".pdf" else "text/plain")}
                response = requests.post(API_URL, files=files_payload)
            
            if response.status_code == 200:
                file_id = response.json().get('file_id')
                print(f"UPLOAD SUCCESS. File ID: {file_id}")
                
                # Trigger processing
                process_url = f"http://localhost:8000/files/files/{file_id}/process"
                proc_response = requests.post(process_url, json={"extract_entities": True})
                
                if proc_response.status_code == 200:
                    print(f"PROCESSING STARTED. Job ID: {proc_response.json().get('job_id')}")
                else:
                    print(f"PROCESSING FAILED: {proc_response.status_code} - {proc_response.text}")

            else:
                print(f"UPLOAD FAILED: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    ingest_all()
