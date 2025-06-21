#!/usr/bin/env python3
"""
Quick fix script to download required NLTK data
Run this if you get NLTK download errors
"""

import nltk

def download_nltk_data():
    """Download all required NLTK data"""
    print("Downloading NLTK data...")
    
    try:
        nltk.download('punkt', quiet=True)
        print("✓ Downloaded 'punkt'")
    except Exception as e:
        print(f"✗ Failed to download 'punkt': {e}")
    
    try:
        nltk.download('punkt_tab', quiet=True)
        print("✓ Downloaded 'punkt_tab'")
    except Exception as e:
        print(f"✗ Failed to download 'punkt_tab': {e}")
    
    try:
        nltk.download('popular', quiet=True)
        print("✓ Downloaded popular NLTK packages")
    except Exception as e:
        print(f"✗ Failed to download popular packages: {e}")
    
    print("\nNLTK setup complete!")

if __name__ == "__main__":
    download_nltk_data()