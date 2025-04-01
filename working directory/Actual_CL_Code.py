import torch
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler, BertConfig
from sklearn.metrics import accuracy_score, classification_report
import os
import copy

# 1. File Paths
old_model_dir = r"C:\Users\dixit\Desktop\pH_Rainfall_0_344_train_model"
new_dataset_path = r"C:\Users\dixit\Downloads\mois0_hum0_356.csv"
output_dir = r"C:\Users\dixit\Desktop\Updated_Model_600"
old_dataset_path = r"C:\Users\dixit\Downloads\344_dataset_ph0_rainfall0.csv"

# 2. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# 3. Load Model Components
print("\nLoading Previous Model Components...")

# Load tokenizer and config first
tokenizer = BertTokenizer.from_pretrained(old_model_dir)
config = BertConfig.from_pretrained(old_model_dir)
old_label_count = config.num_labels

# Load model with handling for potential size mismatches
model = BertForSequenceClassification.from_pretrained(
    old_model_dir,
    config=config,
    ignore_mismatched_sizes=True
).to(device)

# 4. Load Label Dictionary
label_dict_path = os.path.join(old_model_dir, "label_dict.json")
with open(label_dict_path, "r") as f:
    label_dict = json.load(f)
print("\nLoaded Old Label Dictionary:", label_dict)

# 5. Load and Prepare New Dataset
print("\nLoading and Preparing New Dataset...")
new_data = pd.read_csv(new_dataset_path)
new_data.columns = new_data.columns.str.strip()

# Create text representations
new_data['text'] = new_data.apply(lambda row: (
    f"Temperature: {row['Temperature']}, Humidity: {row['Humidity']}, Moisture: {row['Moisture']}, "
    f"Soil Type: {row['Soil_Type']}, Crop Type: {row['Crop_Type']}, "
    f"N: {row['Nitrogen']}, K: {row['Potassium']}, P: {row['Phosphorous']}, "
    f"pH: {row['pH']}, Rainfall: {row['Rainfall']}"
), axis=1)

# 6. Update Label Dictionary with New Labels
new_labels = new_data['Fertilizer'].unique().tolist()
for lbl in new_labels:
    if lbl not in label_dict:
        label_dict[lbl] = len(label_dict)

# Save updated label dictionary
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
    json.dump(label_dict, f)

# 7. Encode Labels
new_data['label'] = new_data['Fertilizer'].map(label_dict)

# 8. Handle Model Expansion for New Labels
new_label_count = len(label_dict)
if new_label_count > old_label_count:
    print(f"\nExpanding model from {old_label_count} to {new_label_count} labels...")
   
    # Create new config
    new_config = copy.deepcopy(config)
    new_config.num_labels = new_label_count
   
    # Create new model with expanded classifier
    new_model = BertForSequenceClassification(new_config).to(device)
   
    # Copy all parameters except classifier
    for (name, param), (new_name, new_param) in zip(model.named_parameters(), new_model.named_parameters()):
        if 'classifier' not in name:
            new_param.data.copy_(param.data)
   
    # Initialize expanded classifier
    if old_label_count > 0:
        new_model.classifier.weight.data[:old_label_count] = model.classifier.weight.data
        new_model.classifier.bias.data[:old_label_count] = model.classifier.bias.data
   
    model = new_model

# 9. Load Old Dataset for Replay
print("\nLoading Old Dataset for Replay...")
old_data = pd.read_csv(old_dataset_path)
old_data['text'] = old_data.apply(lambda row: (
    f"Soil Type: {row['Soil_Type']}, "
    f"N: {row['Nitrogen']}, P: {row['Phosphorous']}, K: {row['Potassium']}, "
    f"pH: {row['pH']}, Rainfall: {row['Rainfall']}, "
    f"Temperature: {row['Temperature']}, Crop Type: {row['Crop_Type']}"
), axis=1)
old_data['label'] = old_data['Fertilizer'].map(label_dict)

# 10. Dataset Class
class FertilizerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 11. Prepare DataLoaders
batch_size = 4
train_data, val_data = train_test_split(new_data, test_size=0.2, random_state=42)

# Create datasets
train_dataset = FertilizerDataset(train_data['text'], train_data['label'], tokenizer)
val_dataset = FertilizerDataset(val_data['text'], val_data['label'], tokenizer)
old_dataset = FertilizerDataset(old_data['text'], old_data['label'], tokenizer)

# Combine old and new datasets
combined_dataset = ConcatDataset([train_dataset, old_dataset])
train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 12. Training Setup
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 1
num_training_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Early stopping
best_val_loss = float('inf')
patience = 3
patience_counter = 0

# 13. Training Loop

print("\nStarting Training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
       
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
           
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(true_labels, predictions)

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

print("\nTraining complete!")
print(f"Best model saved to: {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)