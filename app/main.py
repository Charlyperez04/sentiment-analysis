from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
import pandas as pd
import re
import io

app = FastAPI()

sentiment_model = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis", device=0)

def clean_text(text):
    text = re.sub(r"http\S+", "", text) 
    text = re.sub(r"[^a-zA-Z\s]", "", text) 
    return text.lower().strip() 

class TextRequest(BaseModel):
    text: str

@app.post("/analyze/")
async def analyze_text(request: TextRequest):
    cleaned_text = clean_text(request.text)
    result = sentiment_model(cleaned_text) 
    return {
        "text": request.text,
        "cleaned_text": cleaned_text,
        "sentiment": result[0]["label"],
        "score": result[0]["score"]
    }

@app.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print(contents.decode("utf-8")[:500]) 
        
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), header=None)
        print(df.shape) 

        expected_columns = ["label", "id", "timestamp", "query", "user", "text"]
        if df.shape[1] == len(expected_columns):
            df.columns = expected_columns
        else:
            raise ValueError(f"El archivo no tiene el número esperado de columnas. Se detectaron {df.shape[1]} columnas.")

        df["cleaned_text"] = df["text"].apply(clean_text)

        batch_size = 64
        results = []
        for i in range(0, len(df), batch_size):
            batch_texts = df["cleaned_text"][i:i + batch_size].tolist()
            batch_results = sentiment_model(batch_texts)
            for result in batch_results:
                results.append((result["label"], result["score"]))

        df["sentiment_label"], df["sentiment_score"] = zip(*results)

        output_csv = io.StringIO()
        df.to_csv(output_csv, index=False)
        output_csv.seek(0)
        return {"status": "success", "data_preview": df.head(5).to_dict(orient="records")}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
