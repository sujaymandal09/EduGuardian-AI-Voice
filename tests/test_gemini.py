"""Test if Gemini API is working"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key: {api_key[:20]}..." if api_key else "❌ NO API KEY!")

if not api_key:
    print("❌ Add GEMINI_API_KEY to .env file")
    print("   Get FREE key: https://aistudio.google.com")
    exit()

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is 2+2? Answer in one word."
    )
    print(f"✅ Gemini Response: {response.text.strip()}")
    print("✅ GEMINI IS WORKING!")
    
except Exception as e:
    print(f"❌ Gemini Error: {e}")