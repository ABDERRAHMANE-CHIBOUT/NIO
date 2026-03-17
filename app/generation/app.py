from openai import OpenAI

client = OpenAI(
    api_key="sk-C2D6kUW2RuEXS-N8dF0d9g",
    base_url="http://app.ai-grid.io:4000/v1"
)

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "user", "content": " how can i do embedding for  more then 10 pages without ocr"}
    ]
)

print(response.choices[0].message.content)