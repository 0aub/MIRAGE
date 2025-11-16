#!/usr/bin/env python3
"""
Script to add relevant Saudi Digital Government web pages to MIRAGE
"""

import requests
import time
from typing import List

API_URL = "http://localhost:8000"

# Relevant Arabic web pages about Saudi Digital Government
URLS_TO_PROCESS = [
    # Saudi Digital Government Authority
    "https://www.my.gov.sa/wps/portal/snp/aboutksa/digitaltransformation",
    "https://www.sdaia.gov.sa/ar/default.aspx",
    "https://www.moc.gov.sa/ar/pages/default.aspx",

    # Vision 2030 Digital Transformation
    "https://www.vision2030.gov.sa/ar/v2030/overview/",
    "https://www.vision2030.gov.sa/ar/v2030/v2030-pillars/",

    # Government Digital Services
    "https://www.my.gov.sa/wps/portal/snp/servicesDirectory",
    "https://www.my.gov.sa/wps/portal/snp/content/about",

    # CITC (Communications and Information Technology Commission)
    "https://www.citc.gov.sa/ar/Pages/default.aspx",
]


def process_url(url: str) -> dict:
    """Process a single URL through MIRAGE"""
    print(f"\nProcessing: {url}")

    try:
        response = requests.post(
            f"{API_URL}/url/process",
            json={"url": url, "use_semantic_chunking": True},
            timeout=300  # 5 minute timeout
        )
        response.raise_for_status()
        result = response.json()

        # Print summary
        doc_id = result.get("document_id", "unknown")
        title = result.get("title", "Unknown")
        entities = result.get("phase2", {}).get("entities_extracted", 0)
        relationships = result.get("phase2", {}).get("relationships_extracted", 0)
        words = result.get("phase1", {}).get("total_words", 0)
        processing_time = result.get("processing_time_seconds", 0)

        print(f"✓ Processed: {title}")
        print(f"  - Document ID: {doc_id}")
        print(f"  - Words: {words:,}")
        print(f"  - Entities: {entities}")
        print(f"  - Relationships: {relationships}")
        print(f"  - Processing time: {processing_time}s")

        return result

    except requests.exceptions.Timeout:
        print(f"✗ Timeout processing {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"✗ Error processing {url}: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None


def main():
    """Process all URLs"""
    print("=" * 70)
    print("Adding Saudi Digital Government Sources to MIRAGE")
    print("=" * 70)

    total_urls = len(URLS_TO_PROCESS)
    successful = 0
    failed = 0

    for idx, url in enumerate(URLS_TO_PROCESS, 1):
        print(f"\n[{idx}/{total_urls}] ", end="")

        result = process_url(url)

        if result:
            successful += 1
        else:
            failed += 1

        # Add delay between requests to avoid overwhelming the system
        if idx < total_urls:
            time.sleep(2)

    # Print summary
    print("\n" + "=" * 70)
    print(f"Summary:")
    print(f"  - Total URLs: {total_urls}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
