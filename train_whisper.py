import torch
from datasets import load_dataset, Audio, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# 1. Load Dataset
# Replace 'path/to/data' with your local folder containing metadata.csv and audio files
# metadata.csv format: file_name,transcription
dataset = load_dataset("audiofolder", data_dir=r"c:\Users\UserAdmin\Downloads\Pythoncode\training_data", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 2. Load Processor and Model
model_id = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_id, language="English", task="transcribe")

# Load in 8-bit to save VRAM; use device_map="auto" for automatic GPU placement
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id, 
    quantization_config=quantization_config, 
    device_map="auto"
)

# 3. Prepare model for PEFT (LoRA)
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none"
)
model = get_peft_model(model, config)

# 4. Data Preparation
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# 5. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so it's ignored by the loss function
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 6. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium-lora-out",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=50,
    max_steps=1000, # Adjust based on dataset size
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="no",
    save_steps=200,
    logging_steps=25,
    report_to=["tensorboard"],
    remove_unused_columns=False,
    label_names=["labels"],
)

# 7. Initialize Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    data_collator=data_collator,
    processing_class=processor.feature_extractor,
)

model.config.use_cache = False 
trainer.train()

# 8. Save the adapter weights and processor
trainer.save_model("./whisper-medium-finetuned")
processor.save_pretrained("./whisper-medium-finetuned")