import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
import pandas as pd
import os


# Title of app
st.title('AI Data Analyst')

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

uploaded_file = st.sidebar.file_uploader("Upload data", type=['csv'])


# area to input your API Key
os.environ['OPENAI_API_KEY'] = st.sidebar.text_input('OpenAI API Key', type='password')


if os.environ['OPENAI_API_KEY'] and uploaded_file:
    # model used
    llm = 'gpt-4o'
    chat = ChatOpenAI(model=llm)
    
    # dataset to use
    @st.cache_resource
    def create_sql_database(uploaded_file):
        df = pd.read_csv(uploaded_file)
        engine = create_engine("sqlite:///airline.db")
        df.to_sql('airline-4',engine,index=False)
        db = SQLDatabase(engine=engine)
    
    prompt = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          "You are an expert data analysis with a PhD in data science. Answer all questions with detail and explain your reasoning.",
        ),
        ('human', '{input}'),
        MessagesPlaceholder("agent_scratchpad"),
      ]
    )
    
    memory = ChatMessageHistory(session_id='test-session')

    db = create_sql_database(uploaded_file)
    
    agent_executor = create_sql_agent(chat, db=db, prompt=prompt, agent_type="openai-tools", verbose=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
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
        with st.chat_message("assistant"):
            stream = agent_with_chat_history.stream({'input':prompt}, config={'configurable': {'session_id': 'test-session'}})
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
