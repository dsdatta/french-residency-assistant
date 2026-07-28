from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("residency_doc.txt", "r", encoding="utf-8") as file:
    document = file.read()

system_prompt = f"""You are a helpful assistant that answers questions ONLY using the document below.to be eligible
If the answer isn't in the document, say so clearly instead of guessing  Document:{document}"""

messages = [{"role": "system", "content": system_prompt}]
print("Residency Document Assistant — ask a question, or type 'quit' to exit.\n")

while True:
    user_input = input("You : ")
    if user_input.lower() in ("exit", "quit"):
        break

    messages.append({"role": "user", "content": user_input})
    stream = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0, stream=True
    )

    print("assistant :", end="", flush=True)

    reply = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            print(text, end="", flush=True)
            reply += text

    print("\n")
    messages.append({"role": "assistant", "content": reply})
