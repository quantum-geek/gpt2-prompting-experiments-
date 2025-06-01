from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Text generation function
def generate_response(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Try a zero-shot example
prompt = "Translate this English sentence to French: I love pizza."
print("Zero-shot:\n", generate_response(prompt))

# Few-shot example
prompt = """Translate English to French:
English: Hello, how are you?
French: Bonjour, comment ça va?

English: I love pizza.
French:"""
print("\nFew-shot:\n", generate_response(prompt))

# Chain-of-thought example
prompt = "Question: If John has 3 apples and buys 2 more, how many does he have? Let's think step by step:"
print("\nChain-of-Thought:\n", generate_response(prompt))
