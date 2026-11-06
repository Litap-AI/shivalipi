from datasets import concatenate_datasets, load_dataset


def load_modi_data(
    use_synthetic: bool = True, use_real: bool = True, synthetic_view: str = "image2"
):
    """
    Standardizes Modi datasets into uniform {'image': PIL.Image, 'text': str} records.

    :param use_synthetic: Loads historyHulk/SynthMoDe
    :param use_real: Loads historyHulk/MoDeTrans
    :param synthetic_view: Choose 'image1' (clean) or 'image2' (augmented)
    """
    datasets_to_combine = []

    if use_synthetic:
        synth = load_dataset("historyHulk/SynthMoDe", split="train")

        # Map image1/image2 -> image
        def map_synth(example):
            img = example[synthetic_view]
            if img.mode != "RGB":
                img = img.convert("RGB")
            return {"image": img, "text": example["text"]}

        synth_standardized = synth.map(
            map_synth,
            remove_columns=[col for col in synth.column_names if col != "text"],
        )
        datasets_to_combine.append(synth_standardized)

    if use_real:
        real = load_dataset("historyHulk/MoDeTrans", split="train")

        def map_real(example):
            img = example["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            return {"image": img, "text": example["text"]}

        real_standardized = real.map(
            map_real,
            remove_columns=[
                col for col in real.column_names if col not in ["image", "text"]
            ],
        )
        datasets_to_combine.append(real_standardized)

    if not datasets_to_combine:
        raise ValueError("At least one of use_synthetic or use_real must be True.")

    return concatenate_datasets(datasets_to_combine)
