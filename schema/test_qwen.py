
import os
from openai import OpenAI

# Connect to the Evidence Jam model server
client = OpenAI(
    base_url=os.environ["BRINE_URL"],
    api_key=os.environ["BRINE_API_KEY"]
)

# Send a test message to Qwen
response = client.chat.completions.create(
    model=os.environ["BRINE_MODEL"],
    messages=[
        {
            "role": "system",
            "content": "You are a helpful scientific reasoning assistant."
        },
        {
            "role": "user",
            "content": "Explain the difference between correlation and causation in two sentences."
        }
    ],
    max_tokens=300
)

# Display Qwen's response
print(response.choices[0].message.content)
