from huggingface_hub import InferenceClient
import streamlit as st
import pandas as pd
from googletrans import Translator
import re
import string
import os
import openai
from llama_index.llms.azure_openai import AzureOpenAI
"""
OPENAI_API_TYPE="azure"
OPENAI_API_KEY="e6e399c281c84e9da226cb96d34c2f3a"
OPENAI_API_BASE="https://madhaviopenai1.openai.azure.com/openai/deployments/madhavi/chat/completions?api-version=2024-02-15-preview"
OPENAI_API_VERSION="2024-02-15-preview"
os.environ["AZURE_OPENAI_API_KEY"]=OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"]=OPENAI_API_BASE
os.environ["OPENAI_API_VERSION"]=OPENAI_API_VERSION
"""
llm = AzureOpenAI(
    engine="madhavi",
    model="gpt-4o",
    temperature=0.0,
    azure_endpoint= "https://madhaviopenai1.openai.azure.com",
    api_key="e6e399c281c84e9da226cb96d34c2f3a",
    api_version="2024-02-15-preview",
)

# Function to convert each row in the dataframe
def convert(row):
    s = row['Pin of Interest']
    v = row[s]
    return f"Force {v} on {s} pin and measure the voltage on the same {s} pin with SPU with range of {row['Lower Limit']} and {row['Upper Limit']}."

# Function to clean text
def clean_text(text):
    printable_text = ''.join(char for char in text if char in string.printable)  # Remove non-printable characters
    cleaned_text = re.sub(r'[^\x00-\x7F]', '', printable_text)  # Remove non-ASCII characters
    return cleaned_text

# Function to translate text to English
def translate_to_english(text, src_lang='auto'):
    translator = Translator()
    translation = translator.translate(text, src=src_lang, dest='en')
    return translation.text

# Initialize Hugging Face clients
vishesh_client = InferenceClient("imvishesh007/gemma-Code-Instruct-Finetune-test",token="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")
madhavi_client = InferenceClient("imvishesh007/gemma-Code-Instruct-Finetune-test",token="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")
rakesh_client = InferenceClient("bandi333/gemma-Code-Instruct-Finetune-test-v0.0",token="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")
models = {
    "vishesh_client": vishesh_client,
    "rakesh_client": rakesh_client,      # Add your token and model for rakesh_client if needed
    "madhavi_client": madhavi_client      # Add your token and model for madhavi_client if needed
}
 

def process_client(client, df):
    x = ""
    
    for i in range(df.shape[0]):
        z = st.checkbox(df['english sentence'][i])
        if z:
            response = llm.complete("df['english sentence'][i]")
            x += response
            """
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Ensure the model name is correct
                messages=[{"role": "user", "content": df['english sentence'][i]}],
            )
            x += response.choices[0].message['content']
            #for message in client.chat_completion(messages=[{"role": "user", "content": df['english sentence'][i]}], max_tokens=500, stream=True):
                #print(message.choices[0].delta.content, end="")
                #x += message.choices[0].delta.content
                """
        return x

def main():
    st.set_page_config(layout="wide", page_title="MODELS")

    # Sidebar UI for uploading file and selecting model
    st.sidebar.title("Model Selection and File Upload")
    uploaded_f = st.sidebar.file_uploader("Upload your Excel file", type=["csv"], key="testcase")
    selected_model = st.sidebar.selectbox("Choose your model", options=list(models.keys()))
    
        
        

    if uploaded_f is not None:
        try:
            df = pd.read_csv(uploaded_f)

            # Display original dataframe
            st.subheader("Original Test Case File")
            st.dataframe(df)

            # Convert dataframe to English sentences
            df['english sentence'] = df.apply(convert, axis=1)

            # Display dataframe with English conversion
            st.subheader("Dataframe with English Conversion")
            st.dataframe(df['english sentence'])

            # Add prefix to English sentences
            promtg = "code for the given requirement using customlibrary in cpp for the pin configuration test case"
            df['english sentence'] = df['english sentence'].apply(lambda x: promtg + x)

            # Process selected model
            if selected_model in models and models[selected_model] is not None:
                st.subheader("Interact with Hugging Face Model")
                x = process_client(models[selected_model], df)
            else:
                st.warning(f"Model {selected_model} is not configured or available.")

            # Translate and clean the final output
            x = translate_to_english(x)
            x = clean_text(x)

            # Display final translated and cleaned output
            st.subheader("Final Translated and Cleaned Output")
            st.write(x)

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

if __name__ == '__main__':
    main()
