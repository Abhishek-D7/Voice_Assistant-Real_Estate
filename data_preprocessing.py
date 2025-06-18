# libraries

import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import os

# Load your OpenAI key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load dataset
df = pd.read_csv("real_estate_data.csv")

# Combine relevant columns into single text
df = df.dropna(subset=["Name", "Location", "Price", "Description"])
df["text"] = (
    "Property: " + df["Name"] + ". "
    + df["Property Title"] + ". "
    + "Located in " + df["Location"] + ". "
    + "Total area: " + df["Total Area"].astype(str) + ", "
    + "Price: " + df["Price"].astype(str) + ". "
    + "Baths: " + df["Baths"].astype(str) + ", "
    + "Balcony: " + df["Balcony"].astype(str) + ". "
    + "Description: " + df["Description"]
)

# Create LangChain Documents
docs = [Document(page_content=text) for text in df["text"].tolist()]

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Embed and store in FAISS
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Save to disk
vectorstore.save_local("db/real_estate_index")

print("✅ Preprocessing complete. Vectorstore saved to db/real_estate_index")
print("Files saved at:", os.listdir("db/real_estate_index"))

