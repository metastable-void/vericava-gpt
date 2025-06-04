import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model="vericava/gpt2-medium-vericava-posts-v3", device=device
)

try:
    while True:
        user_input = input("> ").strip()
        print(pipe(user_input, num_return_sequences=1)[0]["generated_text"])
except EOFError:
    print("Exiting.")
