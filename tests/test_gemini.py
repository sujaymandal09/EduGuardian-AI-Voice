"""Test if Groq API is working"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key: {api_key[:20]}..." if api_key else "NO API KEY!")

if not api_key:
    print("Add GROQ_API_KEY to .env file")
    print("Get FREE key: https://console.groq.com")
    exit()

try:
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}]
    )

    print(f"Groq Response: {response.choices[0].message.content.strip()}")
    print("GROQ IS WORKING!")

except Exception as e:
    print(f"Groq Error: {e}")
