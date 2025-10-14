#!/usr/bin/env python3
"""Test OpenRouter API connection"""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/enrico/projects/battalion-platform/.env')

api_key = os.getenv('OPENROUTER_API_KEY')
print(f"API Key found: {'Yes' if api_key else 'No'}")

if api_key:
    print("\nTesting OpenRouter API...")
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o",  # Use GPT-4o as GPT-5 may not exist yet
                "messages": [
                    {"role": "user", "content": "Reply with just: API working"}
                ],
                "max_tokens": 10
            },
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")