import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
	print("Error: API Key not found in .env file!")
else:
	try:
		genai.configure(api_key=api_key)
		print(f"Checking key: {api_key[:10]}...")

		print("\n--- Available Models for you ---")
		found = False
		for m in genai.list_models():
			if 'generateContent' in m.supported_generation_methods:
				print(f"- {m.name}")
				found = True

		if not found:
			print("No models found! Your key might be restricted or billing is required.")

	except Exception as e:
		print(f"\nError: {e}")