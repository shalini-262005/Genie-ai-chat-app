from flask import Flask, request, jsonify, render_template, send_from_directory
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Please set your GOOGLE_API_KEY in the .env file")

# Configure Gemini via google-generativeai
genai.configure(api_key=api_key)

# Flask setup
app = Flask(__name__, static_folder='genie', template_folder='genie')

# Initialize Gemini chat model
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest", generation_config={
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048
})
gemini_convo = gemini_model.start_chat(history=[])

# Global PDF text holder
pdf_text = ""

# ====== Utility Functions ======

def get_pdf_text(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def count_words(text):
    return len(text.split())

def sum_numbers_in_column(text):
    numbers = re.findall(r'\b\d+\b', text)
    return sum(map(int, numbers))

# ====== Conversational QA Chain ======

def get_conversational_chain():
    prompt = ChatPromptTemplate.from_template("""
    Use the following conversation history and context to answer the question. 
    If the answer is not in the context, say "Answer is not available in the context."
    
    Chat History: {chat_history}
    Context: {context}
    Question: {question}
    
    Detailed answer:
    """)
    
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)

    # Chain steps
    chain = (
        RunnableParallel({
            "context": lambda x: "\n\n".join([doc.page_content for doc in x["input_documents"]]),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        })
        | prompt
        | llm
    )
    
    return chain

# ====== Routes ======

@app.route('/')
def home():
    return render_template('Ghome.html')

@app.route('/eg.html')
def eg():
    return render_template('eg.html')

@app.route('/Test.html')
def test():
    return render_template('Test.html')

@app.route('/project.html')
def project():
    return render_template('project.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global pdf_text
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.lower().endswith('.pdf'):
        try:
            pdf_text = get_pdf_text(file)
            chunks = get_text_chunks(pdf_text)
            get_vector_store(chunks)
            return jsonify({'message': 'PDF processed successfully'})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'})

@app.route('/chat', methods=['POST'])
def chat():
    global pdf_text
    data = request.get_json()
    user_message = data['message']
    chat_type = data.get('type', 'gemini')

    try:
        if chat_type == 'gemini':
            gemini_convo.send_message(user_message)
            return jsonify({'response': gemini_convo.last.text})
        
        elif chat_type == 'pdf':
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = vector_store.similarity_search(user_message)

            chain = get_conversational_chain()
            response = chain.invoke({
                "input_documents": docs,
                "question": user_message,
                "chat_history": []
            })

            answer = response.content if hasattr(response, "content") else str(response)

            if "word count" in user_message.lower():
                answer += f"\n\nWord count: {count_words(pdf_text)}"
            if "sum of numbers" in user_message.lower():
                answer += f"\n\nSum of numbers: {sum_numbers_in_column(pdf_text)}"

            return jsonify({'response': answer})
        else:
            return jsonify({'error': 'Invalid chat type'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Serve static files (e.g. CSS/JS/assets)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
