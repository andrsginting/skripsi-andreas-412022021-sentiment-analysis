# sentiment/model_loader.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model_and_tokenizer(model_name="mdhugol/indonesia-bert-sentiment-classification"):
    print(f"[INFO] Loading fine-tuned sentiment model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    print(f"[INFO] Label mapping: {id2label}")
    return tokenizer, model, device, id2label
