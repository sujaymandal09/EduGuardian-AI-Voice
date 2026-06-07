"""Test if Gemini API is working - NEW VERSION"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ NO GEMINI_API_KEY in .env file!")
    print("   Get FREE key: https://aistudio.google.com")
    exit()

print(f"API Key found: {api_key[:15]}...")

try:
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="What is 2+2? Answer in one word."
    )
    
    print(f"✅ Gemini says: {response.text}")
    print("✅ GEMINI IS WORKING!")
    
except Exception as e:
    print(f"❌ Error: {e}")