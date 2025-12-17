import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="how was the messi tour in india?.... rate it ",
    config={
        "tools": [
            {
                "google_search": {}
            }
        ]
    }
)

print("\n=== ANSWER ===\n")
print(response.text)

# print("\n=== GROUNDING METADATA ===\n")
# for c in response.candidates:
#     print(c.grounding_metadata)
