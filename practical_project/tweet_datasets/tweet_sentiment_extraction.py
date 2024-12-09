import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
import evaluate

# Load data from CSV
def load_csv_data(file_path, split):
    data = pd.read_csv(file_path)
    return data

# Load the datasets
data = load_csv_data("./tweet-sentiment-extraction/train.csv", "train")
test_data = load_csv_data("./tweet-sentiment-extraction/test.csv", "test")
data.dropna(inplace=True) #origonal data is modified
test_data.dropna(inplace=True)

# Map labels to integers
label_mapping = {"neutral": 0, "positive": 1, "negative": 2}
data["sentiment"] = data["sentiment"].map(label_mapping)
test_data["sentiment"] = test_data["sentiment"].map(label_mapping)

# Select necessary columns
data = data[["text", "sentiment"]]
test_data = test_data[["text", "sentiment"]]

# Split into train and validation sets (80% train, 20% validation)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Ensure the splits are DataFrames
train_data = pd.DataFrame(train_data)
val_data = pd.DataFrame(val_data)
test_data = pd.DataFrame(test_data)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

# Combine datasets into a DatasetDict
dataset = DatasetDict({
    "train": train_dataset, 
    "validation": val_dataset, 
    "test": test_dataset
    })

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize function
def tokenize_function(examples):
    #print(examples["text"])
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True)
    # Add labels to the tokenized output
    tokens["labels"] = examples["sentiment"]
    return tokens

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare datasets for PyTorch
train_dataset = tokenized_datasets["train"].shuffle(seed=42)
val_dataset = tokenized_datasets["validation"].shuffle(seed=42)
test_dataset = tokenized_datasets["test"].shuffle(seed=42)



# Initialize the BERT model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=len(label_mapping))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Define metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on the test set
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Accuracy:", test_results["eval_accuracy"])

# Save the model and tokenizer
model.save_pretrained("./sentiment-analysis-model")
tokenizer.save_pretrained("./sentiment-analysis-model")
