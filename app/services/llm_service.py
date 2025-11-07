import os
from openai import OpenAI

# Client will be created once and reused
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_chat_response(messages, model="gpt-4o-mini", temperature=0.3):
    """Call OpenAI API and return the generated response."""
    formatted = [{"role": m.role, "content": m.content} for m in messages]

    response = client.chat.completions.create(
        model=model,
        messages=formatted,
        temperature=temperature
    )
    return response.choices[0].message["content"]
