import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

num_classes = 8
label_mapping = torch.load('sentiment_mapping.pt')


def predict_sentiment(sentence):
    model_path = 'model/sentiment-analysis-model.pt'
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02-twitter",
                                                               num_labels=num_classes)
    model.load_state_dict(model_state_dict)
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")
    encoded_input = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    outputs = model(**encoded_input)
    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1).tolist()
    predicted_label_index = predicted_labels[0]
    predicted_label_text = label_mapping[predicted_label_index]
    print(f"Predicted Label: {predicted_label_text}")
    return predicted_label_text


predict_sentiment("هو الميدترم بكره بجد؟؟؟؟")
