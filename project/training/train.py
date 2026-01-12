import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import random


model_name = "google/t5-efficient-mini"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model and tokenizer
print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float32)

print(f'Model on device: {model.device}')
print(f'Model using dtype: {model.dtype}')


# dataset
print("Loading datasets...")

train_dataset = load_dataset('csv', data_files='datasets/dataset_a/train.csv')
valid_dataset = load_dataset('csv', data_files='datasets/dataset_a/valid.csv')

print(f'Number of samples in training: {len(train_dataset["train"])}')
print(f'Number of samples in validation: {len(valid_dataset["train"])}')

sample_train = train_dataset['train'][0]
sample_valid = valid_dataset['train'][0]

print(f"Sample training input: {sample_train['english']}")
print(f"Sample training output: {sample_train['italian']}")
print(f"Sample validation input: {sample_valid['english']}")
print(f"Sample validation output: {sample_valid['italian']}")


# preprocessing
print("Tokenizing datasets...")

def preprocess_function(examples):
    inputs = [ex for ex in examples["english"]]
    targets = [ex for ex in examples["italian"]]
    
    # tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# tokenize the training and validation datasets
tokenized_train = train_dataset['train'].map(preprocess_function, batched=True)
tokenized_valid = valid_dataset['train'].map(preprocess_function, batched=True)

# training config
print("Setting training configuration...")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-custom-dataset",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # TODO add custom metric
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    logging_steps=50,
    # uncomment in case i decide to use other metrics for loss (like BLEU)
    # predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# start training
print("Starting training...")
trainer.train()

# Save the final model
trainer.save_model("./final_model_custom")
tokenizer.save_pretrained("./final_model_custom")
print("Model saved to ./final_model_custom")