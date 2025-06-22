# 🧠 Genie AI Chat app

A Flask-based web app that allows users to upload PDF documents and ask questions, powered by Google Gemini (Generative AI) and LangChain.

---

## 🚀 Features

- Upload PDFs and extract text
- Semantic search and QA using Google Gemini (1.5 Flash)
- Custom embeddings with FAISS and Gemini Embedding API
- Word count and numerical sum support
- Simple web UI (`genie/` templates)

---

## 🛠️ Setup Instructions

### 1. Clone the repository or copy the project files
```bash
git clone <your-repo-url>
cd <your-project-directory>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install required dependencies

```bash
pip install flask python-dotenv google-generativeai langchain langchain-google-genai PyPDF2 faiss-cpu
```

### 4. Set your environment variable

Create a .env file in the root directory and add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### Run the App

```
python app.py
```

### File Structure

```
.
├── app.py                  # Main Flask application
├── genie/                  # Folder containing HTML templates
│   ├── Ghome.html
│   ├── Test.html
│   ├── eg.html
│   └── project.html
├── faiss_index/            # Saved vector store (auto-generated)
├── .env                    # Your API key goes here
```

### Example Questions

"What is the summary of this PDF?"

"What is the word count?"

"Give the sum of numbers in the document."

### Notes

The app uses Gemini 1.5 Flash for both chat and embeddings.

Vector search uses FAISS and is saved in faiss_index.
