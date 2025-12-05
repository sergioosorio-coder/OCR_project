"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
import json
import torch
import evaluate
import numpy as np 

class NotebooksDataset(Dataset):
    def __init__(
    self,
    samples: List[Dict[str, Any]],
    processor: TrOCRProcessor,
    max_target_length: int,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = sample["image"]
        text = sample["text"]

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values.squeeze(0),
            "labels": labels.squeeze(0),
        }

def load_trocr_model_and_processor(
    pretrained_model_name: str,
) -> Dict[str, Any]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(pretrained_model_name)
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name).to(device)

    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "bias" in name:
            p.requires_grad = True


    return {
        "processor": processor,
        "model": model,
    }

def build_trocr_datasets(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    processor: TrOCRProcessor,
    max_target_length: int,
) -> Dict[str, Any]:
    
    train_ds = NotebooksDataset(train_samples, processor, max_target_length)
    val_ds = NotebooksDataset(val_samples, processor, max_target_length)
    return {"train_dataset": train_ds, "val_dataset": val_ds}

def train_trocr_model(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args_dict: Dict[str, Any],
) -> Dict[str, Any]:

    cer_metric = evaluate.load('cer')
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id    
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}
    
    args = Seq2SeqTrainingArguments(**training_args_dict)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        tokenizer=processor.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics = compute_metrics,
        data_collator=default_data_collator,
    )  
    
    trainer.train()
    metrics = trainer.evaluate()
    return {
        "trained_model": model,
        "metrics": metrics,
    }

def save_trocr_artifacts(
    trained_model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    metrics: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # modelo + processor en formato Hugging Face
    trained_model.save_pretrained(out)
    processor.save_pretrained(out)

    # métricas a JSON
    metrics_path = out / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_dir": str(out),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }