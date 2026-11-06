"""
Shivalipi Model Training Pipeline
Fine-tunes a Vision-Encoder-Decoder model to transcribe Modi Lipi to Devanagari.
Author: Rohit Manikrao Patil
"""

from dataclasses import dataclass
from typing import Any

import torch  # type: ignore[import-not-found]
from transformers import (  # type: ignore[import-not-found]
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

from shivalipi.data import load_modi_data
from shivalipi.metrics import compute_cer, compute_wer


@dataclass
class ModiDataCollator:
    """Collates batch samples with image pixel values and padded token labels."""

    processor: ViTImageProcessor
    tokenizer: Any

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [item["image"].convert("RGB") for item in batch]
        texts = [item["text"] for item in batch]

        # Extract patch pixel values
        pixel_values = self.processor(images, return_tensors="pt").pixel_values

        # Tokenize target Devanagari text
        labels = self.tokenizer(
            texts,
            padding="longest",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # Ignore padding tokens in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


def train():
    # 1. Hardware Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # 2. Model & Tokenizer Backbones
    encoder_id = "google/vit-base-patch16-224-in21k"
    decoder_id = "ai4bharat/indic-bert"

    print("Initializing Image Processor & Indic Tokenizer...")
    image_processor = ViTImageProcessor.from_pretrained(encoder_id)
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)

    # 3. Vision-Encoder-Decoder Initialization
    print("Building VisionEncoderDecoder architecture...")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_id, decoder_id
    )

    # Configure special generation tokens
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 1.0
    model.config.num_beams = 4

    # 4. Load & Split Datasets
    print("Loading SynthMoDe and MoDeTrans datasets...")
    full_dataset = load_modi_data(use_synthetic=True, use_real=True)
    splits = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # 5. Metric Computation Hook
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id to decode
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cer = compute_cer(pred_str, label_str)
        wer = compute_wer(pred_str, label_str)
        return {"cer": cer, "wer": wer}

    # 6. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./shivalipi_checkpoints",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=200,
        warmup_steps=100,
        learning_rate=5e-5,
        num_train_epochs=5,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,  # Set True when ready to push to your HF repo
    )

    # 7. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ModiDataCollator(processor=image_processor, tokenizer=tokenizer),
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Saving fine-tuned Shivalipi model...")
    model.save_pretrained("./shivalipi_model")
    tokenizer.save_pretrained("./shivalipi_model")
    image_processor.save_pretrained("./shivalipi_model")
    print("Training complete!")


if __name__ == "__main__":
    train()
