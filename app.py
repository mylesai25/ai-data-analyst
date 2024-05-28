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

# FUNCTIONS
def extract_graphs(content):
  # takes graph from content object
  # returns a list of images to display
  return [Image.open(BytesIO(client.files.content(item.image_file.file_id).read())) for item in content if item.type == 'image_file']

def get_message_text(content):
  # gets text from content object
  # returns text to display on screen
  return content[-1].text.value

# Title of app
st.title('AI Data Analyst')

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

uploaded_file = st.sidebar.file_uploader("Upload data", type=['csv'])

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
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
  
    @st.cache_resource
    def create_file(uploaded_file):
      file = client.files.create(
        file=open(uploaded_file.read(), "rb"),
        purpose='assistants'
      )
      return file
    create_file(uploaded_file)

    
    @st.cache_resource
    def create_message_thread():
        return client.beta.threads.create()

    thread = create_message_thread()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask questions about your data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.spinner(text='Thinking'):
            thread_message = client.beta.threads.messages.create(
              thread.id,
              role="user",
              content=prompt,
              attachments=[
                {
                    "file_id": file.id,
                    "tools": [{"type": "code_interpreter"}]
                }
            ])
            run = client.beta.threads.runs.create_and_poll(
                thread_id = thread.id,
                assistant_id=assistant.id,
                )
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(
                      thread_id = thread.id
                  )
                content = messages.data[0].content
            else:
                st.write(run.status)
            
        with st.chat_message("assistant"):
            text = get_message_text(content)
            plots = extract_graphs(content)
            st.write(text)
            for plot in plots:
                st.write(plot)
        st.session_state.messages.append({"role": "assistant", "content": text})
