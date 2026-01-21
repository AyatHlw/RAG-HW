import google.generativeai as genai
import os
from dotenv import load_dotenv

# تحميل المفتاح
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
	print("Error: API Key not found in .env file!")
else:
	try:
		genai.configure(api_key=api_key)
		print(f"Checking key: {api_key[:10]}...")  # يطبع أول جزء للتأكد

		print("\n--- Available Models for you ---")
		found = False
		for m in genai.list_models():
			# نبحث فقط عن الموديلات التي تدعم توليد النصوص
			if 'generateContent' in m.supported_generation_methods:
				print(f"- {m.name}")
				found = True

		if not found:
			print("No models found! Your key might be restricted or billing is required.")

	except Exception as e:
		print(f"\nError: {e}")