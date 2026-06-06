"""Test one Exotel call"""
import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

SID = "siliguricollege1"
KEY = os.getenv("EXOTEL_API_KEY")
TOKEN = os.getenv("EXOTEL_API_TOKEN")
FROM ="09513886363"
TO = input("Enter your phone (10 digits): ").strip()

print(f"\n📞 Calling {TO} from {FROM}...")

url = f"https://api.exotel.com/v1/Accounts/{SID}/Calls/connect.json"
auth = HTTPBasicAuth(KEY, TOKEN)

data = {
    "From": FROM,
    "To": TO,
    "CallerId": "SiliguriCol",
    "CallType": "trans"
}

response = requests.post(url, data=data, auth=auth, timeout=30)

print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code in [200, 201]:
    print("\n✅ SUCCESS! Phone should ring!")
else:
    print(f"\n❌ Failed with {response.status_code}")
    print("Check your API Key and Token in .env")