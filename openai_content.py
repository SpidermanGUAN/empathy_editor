from openai import OpenAI
client = OpenAI(api_key="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")

content = client.files.content("file-lqLFbE8J7LEcMw4gKz8bh58u")
