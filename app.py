# Requirements: faster-whisper, openai, gradio, transformers, torchaudio,
# langchain, langchain-community, langchain-openai, faiss-cpu, datasets, pandas

import gradio as gr
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
import os
import traceback
import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import numpy as np

# SETUP 
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Warning: OPENAI_API_KEY environment variable not set. RAG features will fail.")

# STT model
ASR_MODEL_SIZE = "base"
print(f"Using ASR model: {ASR_MODEL_SIZE}")
asr_model = WhisperModel(ASR_MODEL_SIZE, device="cpu", compute_type="int8")


# TTS models
tts_models = {
    "en": pipeline("text-to-speech", model="facebook/mms-tts-eng", device="cpu")
}


# Setup LangChain LLM with memory
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# LOAD OR BUILD VECTORSTORE
INDEX_PATH = "real_estate_index"
DATA_PATH = "real_estate_data.csv"

if not os.path.exists(DATA_PATH):
    print(f"'{DATA_PATH}' not found. Creating a dummy CSV file for demonstration.")
    dummy_data = {
        "Property Title": ["Modern Downtown Apartment", "Cozy Suburban House", "Luxury Villa with Pool", "Affordable Studio Near Campus"],
        "Description": ["A beautiful apartment in the heart of the city, close to all amenities.", "A charming house with a large garden, perfect for families.", "An exclusive villa with a private pool and stunning views.", "A compact and affordable studio, ideal for students."],
        "Price": ["$500,000", "$750,000", "$2,000,000", "$250,000"],
        "Location": ["Downtown", "Suburbia", "Hills", "University District"],
        "Total_Area": ["1200 sqft", "2500 sqft", "5000 sqft", "500 sqft"]
    }
    pd.DataFrame(dummy_data).to_csv(DATA_PATH, index=False)


if os.path.exists(INDEX_PATH):
    print("Loading existing vector store...")
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        OpenAIEmbeddings(api_key=openai_api_key),
        allow_dangerous_deserialization=True
    )
else:
    if not openai_api_key:
        print("Cannot build vector store without OPENAI_API_KEY.")
        vectorstore = None
    else:
        print("Building vector store from scratch...")
        df = pd.read_csv(DATA_PATH)
        df.dropna(subset=["Property Title", "Description"], inplace=True)
        docs = []
        for _, row in df.iterrows():
            content = f"Title: {row['Property Title']}\nPrice: {row['Price']}\nLocation: {row['Location']}\nArea: {row['Total_Area']}\nDescription: {row['Description']}"
            docs.append(Document(page_content=content))
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)
        print("Vector store built and saved.")

if vectorstore:
    retriever = vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
else:
    qa_chain = None
    print("Warning: RAG chain not initialized due to missing vector store or API key.")

# FUNCTIONS
def transcribe_audio(audio_path):
    if audio_path is None:
        return "[Error: No audio provided.]"
    try:
        # Transcribe directly from the audio file path provided by Gradio.
        # This is simpler and more robust than manual audio loading.
        segments, info = asr_model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True # Voice Activity Detection can help filter out silence
        )
        print(f"Detected language: {info.language} with probability {info.language_probability}")
        transcription = " ".join([seg.text for seg in segments])
        return transcription.strip() if transcription else "[Info: No speech detected or empty audio.]"
    except Exception as e:
        print("Transcription error:", e)
        traceback.print_exc()
        return "[Error: Unable to transcribe audio.]"


def ask_openai_with_rag(query):
    if not qa_chain:
        return "[Error: RAG system not available. Please check API key and vector store.]"
    if not query or query.startswith("[Error") or query.startswith("[Info"):
        return "I didn't catch that. Could you please repeat or ask something else?"
    try:
        result = qa_chain.invoke({"question": query})
        return result["answer"]
    except Exception as e:
        print("OpenAI RAG error:", e)
        traceback.print_exc()
        return "[Error: Failed to generate response from RAG system.]"

def synthesize_speech(text):
    if not text or text.startswith("[Error") or text.startswith("[Info"):
        return None, None
    try:
        output = tts_models["en"](text)
        audio = output["audio"]
        sample_rate = output["sampling_rate"]

        if isinstance(audio, np.ndarray):
            if audio.ndim == 2 and audio.shape[0] == 1:
                audio = audio.reshape(-1)
            return audio.astype(np.float32), sample_rate
        return None, None
    except Exception as e:
        print("TTS error:", e)
        traceback.print_exc()
        return None, None

def full_pipeline(audio_path):
    transcription = transcribe_audio(audio_path)

    if transcription.startswith("[Error") or transcription == "[Info: No speech detected or empty audio.]":
        error_reply_text = "Sorry, I couldn't understand that or the audio was empty. Please try again."
        audio_array, sample_rate = synthesize_speech(error_reply_text)
        if audio_array is not None and sample_rate is not None:
            return transcription, error_reply_text, (sample_rate, audio_array)
        else:
            return transcription, error_reply_text, None

    reply_text = ask_openai_with_rag(transcription)
    
    if reply_text.startswith("[Error"):
        audio_array, sample_rate = synthesize_speech("I encountered an issue processing your request.")
    else:
        audio_array, sample_rate = synthesize_speech(reply_text)

    if audio_array is not None and sample_rate is not None:
        return transcription, reply_text, (sample_rate, audio_array)
    else:
        return transcription, reply_text, None


# GRADIO UI 
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 Voice-Based Real Estate Assistant")
    #gr.Markdown(f"Using Whisper model: **{ASR_MODEL_SIZE}**. Ask questions about property listings! Ensure `OPENAI_API_KEY` is set.")

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak your real estate question")

    with gr.Column():
        transcribed_text = gr.Textbox(label="📜 Transcription", interactive=False)
        llm_reply = gr.Textbox(label="🤖 Assistant's Reply", interactive=False)
        audio_output = gr.Audio(label="🔊 Assistant's Voice", autoplay=False, interactive=False)

    submit_btn = gr.Button("Ask")

    submit_btn.click(
        fn=full_pipeline,
        inputs=audio_input,
        outputs=[transcribed_text, llm_reply, audio_output]
    )

if __name__ == "__main__":
    if not openai_api_key:
        print("\n" + "="*50)
        print("WARNING: OPENAI_API_KEY is not set.")
        print("The application will run, but RAG and TTS functionalities might be limited or fail.")
        print("Please set the OPENAI_API_KEY environment variable for full functionality.")
        print("="*50 + "\n")
    demo.launch(debug=True)