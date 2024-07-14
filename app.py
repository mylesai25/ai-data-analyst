import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import pandas as pd
import os
from openai import OpenAI
from io import StringIO
from io import BytesIO
from PIL import Image
from st_audiorec import st_audiorec
from pathlib import Path
from openai import OpenAI
import speech_recognition as sr
import requests
import datetime
from tempfile import NamedTemporaryFile
import base64



# FUNCTIONS
def save_audio_file(audio_bytes, file_extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def response_audio(text):
    summary_response = client.chat.completions.create(
        model='gpt-4o',
        messages =[
        {'role': 'system', 'content': 'You are a helpful assistant who specializes in summarizing text from an AI data analyst. Summarize incoming text into a single paragraph.'},
        {'role': 'user', 'content': text}
        ]
    )
    summary = summary_response.choices[0].message.content
    with NamedTemporaryFile(suffix='.mp3') as temp:
        response = client.audio.speech.create(
            model='tts-1',
            voice='alloy',
            input=summary
        )
        tempname = temp.name
        response.stream_to_file(tempname)
        autoplay_audio(tempname)
    return
    
    
    

def extract_graphs(content):
  # takes graph from content object
  # returns a list of images to display
  return [Image.open(BytesIO(client.files.content(item.image_file.file_id).read())) for item in content if item.type == 'image_file']

def get_message_text(content):
  # gets text from content object
  # returns text to display on screen
  return content[-1].text.value

@st.cache_resource
def create_message_thread():
    return client.beta.threads.create()

@st.cache_resource
def create_file(uploaded_file):
  file = client.files.create(
    file=uploaded_file,
    purpose='assistants'
  )
  return file

# Title of app
st.title('AI Data Analyst')

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if 'audio_text' not in st.session_state:
    st.session_state.audio_text = None

st.sidebar.markdown('Please enter your API key and upload your csv file below to start the AI Data Analyst!') 
uploaded_file = st.sidebar.file_uploader("Upload data", type=['csv'])

st.sidebar.markdown('Turn on Text-To-Speech for the AI Data Analyst to give summaries of its responses via speech!')
speech = st.sidebar.radio('Text-To-Speech?', ['On', 'Off'])


if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.thread = create_message_thread()
    st.session_state.audio_text = None
    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.trace_id = None
    st.session_state.file_name =  None
    st.session_state.page = None
    st.session_state.chat_engine = None



# area to input your API Key
os.environ['OPENAI_API_KEY'] = st.sidebar.text_input('OpenAI API Key', type='password')


if os.environ['OPENAI_API_KEY'] and uploaded_file:
    # model used
    client = OpenAI()
    st.sidebar.markdown('You can also give voice commands to the data analyst by pressing the Start Recoding Button.')
    with st.sidebar.container():
      audio_bytes = st_audiorec()
      if audio_bytes:
        file_name = save_audio_file(audio_bytes, 'wav')
        transcript = client.audio.transcriptions.create(
            model='whisper-1',
            file=open(file_name, 'rb')
        )
        st.write(transcript.text)
        if transcript:
            st.session_state.audio_text = transcript.text
            transcript = None
            audio_bytes = None
      else:
        st.session_state.audio_text = None

    # response.stream_to_file(speech_file_path)
    
    file = create_file(uploaded_file)

    st.session_state.thread = create_message_thread()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message['role'] == 'assistant':
              st.markdown(message['content']['text'])
              for fig in message['content']['plots']:
                st.write(fig)
            else:
              st.markdown(message["content"])
        
    # Accept user input
    if (prompt := st.chat_input("Enter chat prompt here") or st.sidebar.button('Submit Audio')):
        # Add user message to chat history
        if st.session_state.audio_text:
            prompt = st.session_state.audio_text
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.spinner(text='Thinking'):
            if len(st.session_state.messages) < 2:
                thread_message = client.beta.threads.messages.create(
                  st.session_state.thread.id,
                  role="user",
                  content=prompt,
                  attachments=[
                    {
                        "file_id": file.id,
                        "tools": [{"type": "code_interpreter"}]
                    }
                ])
            else:
                thread_message = client.beta.threads.messages.create(
                st.session_state.thread.id,
                role="user",
                content=prompt
                )
            run = client.beta.threads.runs.create_and_poll(
                thread_id = st.session_state.thread.id,
                assistant_id='asst_yiM2UfX3tc2bY9nttV2g7KJi',
                )
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(
                      thread_id = st.session_state.thread.id
                  )
                content = messages.data[0].content
            else:
                st.write(run.status)
            
        with st.chat_message("assistant"):
            text = get_message_text(content)
            plots = extract_graphs(content)
            st.markdown(text)
            for plot in plots:
                st.write(plot)
            if speech == 'On':
                response_audio(text)
        st.session_state.messages.append({"role": "assistant", "content": {'text':text, 'plots': plots}})
        
    
    

