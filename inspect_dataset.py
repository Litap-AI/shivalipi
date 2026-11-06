from datasets import load_dataset

print("Fetching sample from MoDeTrans...")
dataset = load_dataset("historyHulk/MoDeTrans", split="train", streaming=True)
sample = next(iter(dataset))

print("\n--- Sample Keys & Metadata ---")
for key, value in sample.items():
    if key != "image":
        print(f"{key}: {value}")
    else:
        print(
            f"image: PIL.Image format={value.format}, size={value.size}, mode={value.mode}"
        )
