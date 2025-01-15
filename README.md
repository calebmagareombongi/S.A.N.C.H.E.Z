# README.md

## Overview
# S.A.N.C.H.E.Z Virttual Assistant Chatbot

This project includes two main implementations for querying GPT models using text from PDFs or voice input:

1. **Streamlit-based User Interface**:
   - A web application that allows users to upload PDFs or use voice input to query OpenAI's GPT models.
   - Includes integration with the Vosk speech recognition library for voice transcription.

2. **FastAPI-based REST API**:
   - A backend API for querying GPT models using PDF files or direct questions.
   - Provides endpoints for processing PDF uploads and answering questions.

---

## Features

### Streamlit Application
- Upload PDFs and extract text for querying GPT models.
- Select GPT models (`gpt-3.5-turbo`, `gpt-4`), and customize query parameters like `max_tokens` and `temperature`.
- Use voice input, transcribed with Vosk, to ask questions and retrieve answers.
- Provides a user-friendly interface for interacting with GPT models.

### FastAPI API
- Endpoints for:
  - Querying PDFs: Extracts text from uploaded PDF files and answers questions based on the extracted text.
  - Directly asking questions to GPT models.
- Customizable query parameters: `max_tokens`, `temperature`, and model selection.

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- API key for OpenAI services
- Required Python libraries: `streamlit`, `fastapi`, `uvicorn`, `PyPDF2`, `vosk`, `speech_recognition`, and `wave`

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Vosk model** (for the Streamlit app):
   - The application will automatically download the model when needed.

5. **Set up OpenAI API key**:
   - Replace `"your API key"` and `"your_openai_api_key"` with your actual OpenAI API key in the respective code files.

---

## Running the Applications

### Streamlit Application
1. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Open the URL displayed in the terminal (e.g., `http://localhost:8501`) in your browser.
3. Use the interface to upload PDFs or interact with the voice input feature.

### FastAPI API
1. Start the FastAPI server:
   ```bash
   uvicorn fastapi_app:app --reload
   ```
2. Access the API documentation at:
   ```
   http://127.0.0.1:8000/docs
   ```
3. Test the API endpoints using tools like `Postman` or `curl`.

---

## Example Usage

### Streamlit
1. Upload a PDF file.
2. Enter a question related to the content of the PDF.
3. Select a model (`gpt-3.5-turbo` or `gpt-4`).
4. View the extracted context and GPT's response.

### FastAPI
- **Query a PDF**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/query-pdf" \
  -F "file=@example.pdf" \
  -F "question=What is this document about?" \
  -F "model=gpt-4" \
  -F "max_tokens=300" \
  -F "temperature=0.7"
  ```
- **Ask a direct question**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/ask-question" \
  -d "question=Explain quantum mechanics." \
  -d "model=gpt-4" \
  -d "max_tokens=300" \
  -d "temperature=0.7"
  ```

---

## Notes
- Ensure the `vosk-model-en-us-0.22` model is downloaded for voice transcription to work.
- The API key is hardcoded for simplicity; consider using environment variables for better security.

---

## License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.
