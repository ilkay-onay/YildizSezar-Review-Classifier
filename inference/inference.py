from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "/var/home/hpo15/development/python/tez/bert-tabanli/tezv7ultraegitim/final_model_2024-12-16_21-20-10/"
tokenizer = ElectraTokenizer.from_pretrained(model_name)
model = ElectraForSequenceClassification.from_pretrained(model_name)

model.to(device)

def predict_star_rating(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    star_rating = predicted_class + 1  
    return star_rating

review_text = "Biraz gecıkmeli aktarıyor görüntüyü ama onun dışında fiyat performans ürünü."
predicted_rating = predict_star_rating(review_text)
print(f"Predicted Star Rating: {predicted_rating}")
