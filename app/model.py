from transformers import pipeline

sentiment_model = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
result = sentiment_model("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]
