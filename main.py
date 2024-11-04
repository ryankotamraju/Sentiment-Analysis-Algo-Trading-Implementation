import re
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, Trainer, TrainingArguments
from torch.nn.functional import softmax
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from datasets import Dataset, DatasetDict
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
ner_pipeline = pipeline("ner", device="mps")


def clean_text(text) -> str:
    t = text.lower()
    t = re.sub(r'[^a-z\s]', '', t)
    return t


def extract_entities(text):
    entities = ner_pipeline(text)
    return entities


categories = ['rec.autos', 'rec.sport.baseball', 'sci.crypt', 'rec.sport.hockey']
news_train = fetch_20newsgroups(subset='train', categories=categories)

vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(news_train.data)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, news_train.target)


def classify_topic(text):
    text_vector = vectorizer.transform([text])
    topic_index = classifier.predict(text_vector)[0]
    topic = news_train.target_names[topic_index]
    return topic

model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name).to(torch.device("mps"))

train_texts = ["Exceeded sales targets", "Poor quarterly results.", "Product launch success.", "Market share loss."]
train_labels = [1, 0, 1, 0]

dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
dataset = dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    "train": dataset['train'],
    "test": dataset['test']
})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()


def predict_sentiment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_tensors='pt'
    ).to(torch.device("mps"))

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.to("cpu")  # Ensure logits are on CPU
        probabilities = softmax(logits, dim=-1)
        sentiment_score = torch.argmax(probabilities, dim=-1).item()
    return sentiment_score


def make_trading_decision(sentiment_score):
    if sentiment_score >= 4:
        return "Buy"
    elif sentiment_score <= 2:
        return "Sell"
    else:
        return "Hold"


test_article = "The company ABC reported a substantial rise in profits for Q3 2024!"
cleaned_test_article = clean_text(test_article)
predicted_sentiment = predict_sentiment(cleaned_test_article)
print(f"Predicted Sentiment Score: {predicted_sentiment}")

decision = make_trading_decision(predicted_sentiment)
print(f"Trading Decision: {decision}")
