"""Minimal client example for the optional local AgeVision API."""
import os
import requests

api_key = os.environ["AGE_GENDER_API_KEY"]
with open("example.jpg", "rb") as image_file:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/analyze",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"image": ("example.jpg", image_file, "image/jpeg")},
        timeout=120,
    )
response.raise_for_status()
print(response.json())
