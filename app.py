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
    llm = 'gpt-4o'
    chat = ChatOpenAI(model=llm, temperature=0)
    
    # dataset to use
    @st.cache_resource
    def create_df_database(uploaded_file):
        df = pd.read_csv(uploaded_file)
        return df
    
    @st.cache_resource
    def create_chat_agent(df):
        prompt = ChatPromptTemplate.from_messages(
          [
            (
              "system",
              "You are an expert data analysis with a PhD in data science. Answer all questions with detail and explain your reasoning. Use the chat history for context for your next answer.",
            ),
            MessagesPlaceholder("chat_history")
            ('human', 'Answer this query in a detailed and polite manner: {input}'),
            MessagesPlaceholder("agent_scratchpad"),

          ]
        )
        
        memory = ChatMessageHistory(session_id='test-session')
    
        
        
        agent_executor = create_pandas_dataframe_agent(chat, df, prompt=prompt, agent_type='openai-tools', verbose=True)
        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            # This is needed because in most real world scenarios, a session id is needed
            # It isn't really used here because we are using a simple in memory ChatMessageHistory
            lambda session_id: memory,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return agent_with_chat_history
        
    df = create_df_database(uploaded_file)
    agent_with_chat_history = create_chat_agent(df)
    
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
                stream = agent_with_chat_history.invoke({'input':prompt}, config={'configurable': {'session_id': "test-session"}})
        with st.chat_message("assistant"):
                response = stream['output']
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
