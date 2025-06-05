import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model="vericava/gpt2-medium-vericava-posts-v3", device=device
)

try:
    while True:
        user_input = " ".join(input("> ").strip().split("\n"))
        if user_input == "":
            continue

        user_input = user_input if (
            user_input.endswith("。")
            or user_input.endswith("?")
            or user_input.endswith("!")
            or user_input.endswith("？")
            or user_input.endswith("！")
        ) else user_input + "。"

        gen_text = pipe(
            user_input,
            num_return_sequences=1,
            temperature=1.2,
        )[0]["generated_text"]
        gen_text = gen_text[len(user_input):]

        gen_text = gen_text[:gen_text.find("\n")] if "\n" in gen_text else gen_text
        gen_text = gen_text[:(gen_text.rfind("。") + 1)] if "。" in gen_text else gen_text

        print(gen_text)
except EOFError:
    print("\nBye. Exiting.")
except KeyboardInterrupt:
    print("\nInterrupted. Bye.")
