import streamlit as st
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering

# Load TAPAS model
model_name = "google/tapas-base-finetuned-wtq"  # Using a smaller model for efficiency
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name)

st.title("📊 AI-Powered Data Query with TAPAS")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 🔹 Convert all values to string (Fix for TypeError)
    df = df.astype(str)

    # 🔹 Limit the number of rows to 500 (Fix for "Too many rows" error)
    max_rows = 500
    if len(df) > max_rows:
        st.warning(f"⚠️ The dataset is too large. Only the first {max_rows} rows will be used.")
        df = df.head(max_rows)

    st.write("### Preview of the Data:")
    st.write(df.head())

    query = st.text_input("🔍 Ask a question about the table:")

    if st.button("Get Answer"):
        if query:
            try:
                # Convert to TAPAS format
                inputs = tokenizer(table=df, queries=[query], padding="max_length", return_tensors="pt")
                outputs = model(**inputs)

                # Convert logits to predictions
                answers = tokenizer.convert_logits_to_predictions(inputs, outputs.logits, outputs.logits_aggregation)
                st.write("### 🤖 Answer:", answers[0])
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question.")
