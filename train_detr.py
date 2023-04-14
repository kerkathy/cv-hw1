from transformers import DetrFeatureExtractor, TrainingArguments, Trainer, DetrForObjectDetection
from dataset import CocoDetection
from torch.utils.data import DataLoader
from pathlib import Path
# from model import Detr


data_path = Path('hw1_dataset/images')
model_checkpoint = "facebook/detr-resnet-50"
# feature_extractor = DetrFeatureExtractor.from_pretrained(model_checkpoint)
processor = DetrFeatureExtractor.from_pretrained(model_checkpoint)

train_dataset = CocoDetection(img_folder=data_path / 'train', processor=processor)
val_dataset = CocoDetection(img_folder=data_path / 'valid', processor=processor, train=False)

# Try with one example for feature_extractor and see if it works
print("train_dataset: ", train_dataset[0])
print("pixel_values.shape: ", train_dataset[0]["pixel_values"].shape)
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))
print(" ")

train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.collate_fn, batch_size=2)
batch = next(iter(train_dataloader))

id2label = {x["id"]: x["name"] for x in train_dataset.categories}
label2id = {v: k for k, v in id2label.items()}


model = DetrForObjectDetection.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="detr-object-detection-finetuned",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    logging_steps=50,
    save_steps=200,
    load_best_model_at_end=True,
    save_total_limit=2,
)

# Initalize our trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=train_dataset.collate_fn,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
)

checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
max_train_samples = len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()     


