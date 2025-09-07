#!/usr/bin/env python3
"""
Script to set up project directories and initial structure.
"""

import os
from pathlib import Path

def create_directories():
    """Create necessary project directories."""
    directories = [
        "data/train",
        "data/validation", 
        "data/test",
        "models",
        "logs",
        "output",
        "examples",
        "scripts",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create placeholder files
    placeholder_files = [
        "data/train/.gitkeep",
        "data/validation/.gitkeep",
        "data/test/.gitkeep",
        "models/.gitkeep",
        "logs/.gitkeep",
        "output/.gitkeep"
    ]
    
    for file_path in placeholder_files:
        Path(file_path).touch()
        print(f"Created placeholder file: {file_path}")

if __name__ == "__main__":
    print("Setting up project directories...")
    create_directories()
    print("Project setup completed!")
