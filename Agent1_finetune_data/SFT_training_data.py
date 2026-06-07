from datasets import load_dataset

dataset = load_dataset("Estwld/empathetic_dialogues_llm")

def format_with_brackets(example):
    convs = example['conversations']

    # Using f-string to wrap the role in brackets and capitalize it
    formatted_turns = [
        f"[{turn['role'].capitalize()}]: {turn['content']}"
        for turn in convs
    ]

    # Joining with a space or newline for better readability
    chat_history = " ".join(formatted_turns)

    return {"chat_history": chat_history}

# Apply to dataset
dataset = dataset.map(format_with_brackets)

# Verify the output
print(dataset['train'][0]['chat_history'])