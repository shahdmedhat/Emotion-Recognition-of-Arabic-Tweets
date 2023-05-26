from torch.utils.data import TensorDataset, DataLoader
import ast
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import classification_report

total_df = pd.read_csv('dataset/preprocessed_tweets.csv')
total_df = total_df.drop("ID", axis=1)
# print(df.columns)

df = total_df.head(1000)
num_classes = 8  # Number of sentiment classes

tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")
model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02-twitter",
                                                           num_labels=num_classes)

# Splitting the dataset into training and testing
train_df, test_df = train_test_split(total_df, test_size=0.2, random_state=42)
test_df = test_df.reset_index(drop=True)

# print(train_df.head())
# print("SIZE OF TRAINING DATASET: ",train_df.shape[0])

# print(test_df.head())
# print("SIZE OF TESTING DATASET: ",test_df.shape[0])

string_lists = train_df[' TWEET'].tolist()
train_tokens = [ast.literal_eval(string) for string in string_lists]
# print(train_tokens)

train_input_encodings = []

# Encoding
for tokens in train_tokens:
    sequence = ' '.join(tokens)  # Convert the list of tokens to a single string
    encoding = tokenizer.encode_plus(sequence, padding='max_length', truncation=True, max_length=128,
                                     return_tensors='pt')
    train_input_encodings.append(encoding)

# for input_ids in train_input_encodings:
#     print(input_ids)

train_input_ids = []
train_attention_mask = []

for encoding in train_input_encodings:
    input_ids = encoding["input_ids"]
    train_input_ids.append(input_ids)

    attention_mask = encoding["attention_mask"]
    train_attention_mask.append(attention_mask)

# train_labels = torch.tensor(train_df[" LABEL"].tolist())

label_encoder = LabelEncoder()
train_encoded_labels = label_encoder.fit_transform(train_df[" LABEL"])

# print(train_input_ids)
# print(train_attention_mask)
# print(encoded_labels)

train_input_ids_stacked_tensor = torch.stack(train_input_ids, dim=0)
train_attention_mask_stacked_tensor = torch.stack(train_attention_mask, dim=0)
train_labels_tensor = torch.tensor(train_encoded_labels)

# Creating PyTorch dataset
train_dataset = TensorDataset(train_input_ids_stacked_tensor, train_attention_mask_stacked_tensor, train_labels_tensor)

# --------------------------------------------------------------------------------

test_tokens_lists = test_df[' TWEET'].tolist()
test_tokens = [ast.literal_eval(string) for string in test_tokens_lists]

test_input_encodings = []
for tokens in test_tokens:
    sequence = ' '.join(tokens)
    encoding = tokenizer.encode_plus(sequence, padding='max_length', truncation=True, max_length=128,
                                     return_tensors='pt')
    test_input_encodings.append(encoding)

test_input_ids = []
test_attention_mask = []

for encoding in test_input_encodings:
    input_ids = encoding["input_ids"]
    test_input_ids.append(input_ids)

    attention_mask = encoding["attention_mask"]
    test_attention_mask.append(attention_mask)

test_encoded_labels = label_encoder.fit_transform(test_df[" LABEL"])

test_input_ids_stacked_tensor = torch.stack(test_input_ids, dim=0)
test_attention_mask_stacked_tensor = torch.stack(test_attention_mask, dim=0)
test_labels_tensor = torch.tensor(test_encoded_labels)

test_dataset = TensorDataset(test_input_ids_stacked_tensor, test_attention_mask_stacked_tensor, test_labels_tensor)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# early stopping parameters
best_loss = float('inf')  # best loss = infinity
patience = 3  # no of epochs to wait for improvement
counter = 0  # no of epochs without improvement

label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
torch.save(label_mapping, 'sentiment_mapping.pt')


# Training
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)

        # print(input_ids)

        optimizer.zero_grad()

        # print(input_ids.shape)
        # print(attention_masks.shape)

        input_ids = input_ids.squeeze(dim=1)
        attention_masks = attention_masks.squeeze(dim=1)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_masks)
        logits = outputs.logits

        loss = loss_fn(logits, labels)

        # Backward pass
        loss.backward()

        # Updating weights
        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Step {step}/{len(train_dataloader)} - Loss: {loss.item()}")
            print("-------------------------------------------------------------------------------")

    # average loss per epoch
    avg_loss = total_loss / len(train_dataloader)

    # loss improved?
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
    else:
        counter += 1

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # early stopping if no improvement
    if counter >= patience:
        print("No improvement in loss. Early stopping...")
        break

tweets = test_df[' TWEET'].tolist()

model.eval()

with torch.no_grad():
    total_correct = 0
    total_samples = 0
    predicted_labels = []
    actual_labels = []
    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)

        input_ids = input_ids.squeeze(dim=1)
        attention_masks = attention_masks.squeeze(dim=1)

        batch_tweets = [tweets[i] for i in range(total_samples, total_samples + len(labels))]

        for i in range(len(labels)):
            tweet = batch_tweets[i]

            current_input_ids = input_ids[i].unsqueeze(0)
            current_attention_masks = attention_masks[i].unsqueeze(0)
            current_labels = labels[i].unsqueeze(0)

            outputs = model(input_ids=current_input_ids, attention_mask=current_attention_masks)
            logits = outputs.logits

            predicted_label = torch.argmax(logits, dim=1)

            predicted_labels.append(predicted_label.item())
            actual_labels.append(current_labels.item())

            print(f"Tweet: {tweet}")
            print(f"Predicted Label: {label_mapping[predicted_label.item()]}")
            print(f"Actual Label: {label_mapping[current_labels.item()]}")
            print()

            total_correct += (predicted_label == current_labels).sum().item()
            total_samples += 1

    accuracy = total_correct / total_samples
    f1_score = \
    classification_report(actual_labels, predicted_labels, target_names=list(label_mapping.values()), output_dict=True)[
        'macro avg']['f1-score']
    recall = \
    classification_report(actual_labels, predicted_labels, target_names=list(label_mapping.values()), output_dict=True)[
        'macro avg']['recall']
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Recall: {recall:.4f}")

torch.save(model.state_dict(), 'model/sentiment-analysis-model.pt')
