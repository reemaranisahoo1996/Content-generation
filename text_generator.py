# Install required libraries (uncomment if running for the first time)
# !pip install langchain-huggingface
# !pip install huggingface_hub
# !pip install transformers
# !pip install accelerate
# !pip install bitsandbytes
# !pip install langchain
# !pip install streamlit

# Import necessary libraries
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import login
import os
import streamlit as st
from transformers import pipeline

import os
os

# Set Hugging Face token directly or obtain it from your environment
hf_token = os.environ.get("HF_TOKEN")
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize the Hugging Face LLM endpoint
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=hf_token)

# Streamlit app code starts here

# Sidebar for Hugging Face Token and LLM Model
st.sidebar.title("Model Configuration")
hf_token = st.sidebar.text_input("Enter your Hugging Face Token", type="password", value=hf_token)
llm_model = st.sidebar.text_input("Enter the LLM Model Name", value=repo_id)

# Main interface for text generation
st.title("Text Generation with LLM")
input_text = st.text_area("Enter your prompt:", "Once upon a time...")

# Button to generate text
if st.button("Generate Text"):
    if not hf_token:
        st.error("Please enter your Hugging Face token in the sidebar.")
    else:
        with st.spinner("Generating text..."):
            # Initialize the Hugging Face pipeline for text generation using the token and selected model
            generator = pipeline("text-generation", model=llm_model, use_auth_token=hf_token)
            output = generator(input_text, max_length=100, num_return_seqpramoduences=1)

            # Display the generated text
            st.subheader("Generated Text:")
            st.write(output[0]['generated_text'])

# Note: To run the app, use: streamlit run text_generation_app.py
