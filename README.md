# AI Data Analyst

This Streamlit application uses OpenAI's GPT-4 and other tools to create an interactive AI data analyst. The app allows users to upload data files, generate text or speech inputs, and receive analytical responses, including textual summaries and visualizations.

## Features

- **Data Upload**: Upload CSV files for analysis.
- **Text-to-Speech**: Convert text responses to audio.
- **Speech Recognition**: Record and transcribe audio inputs.
- **Chat Interface**: Interactive chat interface for user queries and responses.
- **Graph Extraction**: Extract and display graphs from AI-generated content.

## Requirements

- Python 3.8 or higher
- Streamlit
- OpenAI API key
- Additional libraries: `langchain_core`, `langchain_community`, `langchain_openai`, `sqlalchemy`, `matplotlib`, `pandas`, `PIL`, `st_audiorec`, `speech_recognition`, `requests`

## Setup

1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up your OpenAI API key**:
    - Open the `.env` file and add your OpenAI API key:
    ```sh
    OPENAI_API_KEY=<your_openai_api_key>
    ```

4. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

## Usage

1. **Upload Data**:
    - Use the sidebar to upload your CSV file.

2. **Text-to-Speech**:
    - Toggle the Text-to-Speech option in the sidebar.

3. **Record Audio**:
    - Use the audio recorder in the sidebar to capture and transcribe speech inputs.

4. **Chat Interface**:
    - Enter your chat prompts in the input box or submit audio recordings.
    - View responses, including text and graphs, directly in the chat interface.

5. **Clear Chat**:
    - Use the "Clear Chat" button in the sidebar to reset the chat history.

## Functions

### `save_audio_file(audio_bytes, file_extension)`
Saves recorded audio to a file with the given extension.

### `autoplay_audio(file_path: str)`
Autoplays the audio from the specified file path in the Streamlit app.

### `response_audio(text)`
Generates an audio response for the given text and plays it back.

### `extract_graphs(content)`
Extracts graph images from AI-generated content.

### `get_message_text(content)`
Extracts text from AI-generated content.

### `create_message_thread()`
Creates a new message thread for the AI conversation.

### `create_file(uploaded_file)`
Uploads the provided file to the AI service for analysis.

## Components

- **`st_audiorec`**: Used for recording audio inputs.
- **`openai`**: OpenAI API client for generating responses.
- **`pandas`**: Data manipulation and analysis.
- **`matplotlib`**: Visualization library for creating plots.

## Notes

- Ensure you have a valid OpenAI API key to use the AI features.
- The app caches resources like message threads and uploaded files for efficiency.
- The chat interface displays both user inputs and AI-generated responses, including any visual content.

## Contributing

Feel free to fork the repository and submit pull requests for improvements or additional features.

## License

This project is licensed under the MIT License.

---

Enjoy your interactive AI data analysis experience!
