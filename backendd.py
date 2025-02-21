import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch

# Load TAPAS model and tokenizer
model_name = "google/tapas-large-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name)

def query_table(file_path: str, query: str = "What is the total sales?"):
    # Read CSV into DataFrame
    df = pd.read_csv(file_path)

    # Tokenize query and table
    inputs = tokenizer(table=df, queries=[query], padding="max_length", return_tensors="pt")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract answer
    predicted_id = outputs.logits.argmax(dim=-1).numpy().flatten().tolist()
    answer = tokenizer.convert_ids_to_tokens(predicted_id, skip_special_tokens=True)

    return {"query": query, "answer": " ".join(answer)}

# Example usage
if __name__ == "__main__":
    file_path = "data.csv"  # Replace with the path to your CSV file
    query = "What is the total sales?"
    result = query_table(file_path, query)
    print(result)
