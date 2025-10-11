import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = """
Generate a complete Tetris game as a single HTML file.
It should include CSS for styling, JavaScript for gameplay logic, and an HTML structure.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an expert web developer."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2,
    max_tokens=2000
)

tetris_code = response.choices[0].message.content
with open("tetris.html", "w", encoding="utf-8") as f:
    f.write(tetris_code)

print("Tetris HTML file generated!")
