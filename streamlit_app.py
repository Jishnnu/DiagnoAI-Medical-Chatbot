# Import necessary libraries
import io
import requests
import nltk
from nltk.tokenize import word_tokenize
import streamlit as st
import speech_recognition as sr
from PIL import Image
from gtts import gTTS
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import JinaChat
from langchain.tools import DuckDuckGoSearchRun

headers = {"Authorization": f"Bearer YOUR_HUGGINGFACE_API_KEY"}
nltk.download("punkt")


# Define a function to split the summarized text into meaningful words
def split_into_meaningful_words(text):
    words = word_tokenize(text)
    meaningful_words = [
        word for word in words if word.isalnum()
    ]  # Keep only alphanumeric words
    return ", ".join(meaningful_words)


# Define a function to summarize the transcribed text
def text_summarization_query(payload):
    API_URL = (
        "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    )
    # data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=payload)
    return response.json()


# Define a function to generate an image of the summarized text
def text_to_image_query(payload):
    API_URL = (
        # Both these models rendered reasonably good images. Choose which works better according to your use case
        "https://api-inference.huggingface.co/models/artificialguybr/IconsRedmond-IconsLoraForSDXL"
        # "https://api-inference.huggingface.co/models/Linaqruf/animagine-xl"
    )
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


# Setup the Streamlit page
st.set_page_config(page_title="DiagnoAI", page_icon="🤖", layout="centered")
st.title("🤖 DiagnoAI : Health first!")

transcribed_text = ""
transcription_response = ""
recognizer = sr.Recognizer()

# Setup the first chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"},
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get the audio input
audio_file = st.file_uploader("Upload your audio file", type="wav")

if audio_file:
    st.audio(audio_file)

    # Transcribe the audio
    with st.spinner("Transcribing... Please wait"):
        with sr.AudioFile(audio_file.name) as source:
            text = recognizer.listen(source=source)
        transcribed_text = recognizer.recognize_google(text, show_all=False)
    st.session_state.messages.append({"role": "user", "content": transcribed_text})
    st.chat_message("user").write(transcribed_text)

    # Setup closed-source JinaChat API. You can replace this with OpenAI or any other chat-based LLM
    chat = JinaChat(
        temperature=0.2,
        streaming=True,
        jinachat_api_key="YOU_API_KEY",
    )

    # Initialize the transcription agent with DuckDuckGo and JinaChat
    transcription_agent = initialize_agent(
        tools=[DuckDuckGoSearchRun(name="Search")],
        llm=chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
    )

    # Initialize the text to speech agent
    output_filename = "Output_Audio.wav"

    # Setup the chat for response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Transcription Chat
        with st.spinner("Loading... Please wait"):
            transcription_response = transcription_agent.run(
                st.session_state.messages, callbacks=[st_cb]
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": transcription_response}
            )
        st.write(transcription_response)

        # Text to Voice Chat
        with st.spinner("Generating voice output... Please wait"):
            speech = gTTS(text=transcription_response, lang="en", slow=False)
            speech_response = speech.save(output_filename)
            st.session_state.messages.append(
                {"role": "assistant", "content": speech_response}
            )
        st.audio(output_filename)

        # Text to Image Chat
        with st.spinner("Generating image output... Please wait"):
            summarized_text = text_summarization_query(
                {
                    "inputs": str(transcription_response)
                    + "-- Please summarize the given text into actionable keywords. Should not exceed 20 words.",
                    "options": {"wait_for_model": True},
                }
            )
            prompt_words = split_into_meaningful_words(str(summarized_text))
            image_bytes = text_to_image_query(
                {
                    "inputs": prompt_words
                    + "1 human, english language, exercise, healthy diet, medicines, vegetables, fruits",
                    "options": {"wait_for_model": True},
                }
            )
            image_response = Image.open(io.BytesIO(image_bytes))
            st.session_state.messages.append(
                {"role": "assistant", "content": image_response}
            )
        st.image(image_response, use_column_width="auto")
