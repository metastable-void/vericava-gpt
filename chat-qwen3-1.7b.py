import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model="vericava/qwen3-1.7b-vericava-posts-v1", device=device
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
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0,
            max_new_tokens=512,
            do_sample=True,
            repetition_penalty=2.0,
        )[0]["generated_text"]
        gen_text = gen_text[len(user_input):]

        gen_text = gen_text[:gen_text.find("\n")] if "\n" in gen_text else gen_text
        gen_text = gen_text[:(gen_text.rfind("。") + 1)] if "。" in gen_text else gen_text

        print(gen_text)
except EOFError:
    print("\nBye. Exiting.")
except KeyboardInterrupt:
    print("\nInterrupted. Bye.")
