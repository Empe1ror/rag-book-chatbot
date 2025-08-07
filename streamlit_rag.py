import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import io

# Import PDF and DOCX readers
try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pandas as pd
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Load environment
load_dotenv()

# Streamlit configuration
st.set_page_config(
    page_title="📚 RAG Book Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-info {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def read_pdf(uploaded_file):
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        raise ImportError("pypdf not installed. Please install it to read PDF files.")
    
    text = ""
    pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
    
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text

def read_docx(uploaded_file):
    """Extract text from DOCX file"""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not installed. Please install it to read DOCX files.")
    
    doc = Document(io.BytesIO(uploaded_file.read()))
    text = ""
    
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    
    return text

def read_excel(uploaded_file):
    """Extract text from Excel file"""
    if not EXCEL_AVAILABLE:
        raise ImportError("pandas not installed. Please install it to read Excel files.")
    
    df = pd.read_excel(io.BytesIO(uploaded_file.read()))
    # Convert DataFrame to text representation
    text = df.to_string(index=False)
    return text

def read_uploaded_file(uploaded_file):
    """Read content from uploaded file based on its type"""
    file_type = uploaded_file.type
    file_name = uploaded_file.name.lower()
    
    try:
        if file_type == "application/pdf" or file_name.endswith('.pdf'):
            return read_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_name.endswith('.docx'):
            return read_docx(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            return read_excel(uploaded_file)
        else:
            # Handle text files
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return content
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_data
def load_book_data(uploaded_file=None, sample_content=None):
    """Load and process the book data (cached for performance)"""
    try:
        if sample_content is not None:
            # Use sample content
            content = sample_content
            docs = [type('TempDoc', (), {'page_content': content, 'metadata': {}})()]
        elif uploaded_file is not None:
            # Load from uploaded file
            content = read_uploaded_file(uploaded_file)
            if content is None:
                return [], 0
            docs = [type('TempDoc', (), {'page_content': content, 'metadata': {}})()]
        else:
            # Load the default book
            loader = TextLoader("book.txt", encoding="utf-8")
            docs = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        # Handle different document types
        if sample_content is not None or uploaded_file is not None:
            chunks = text_splitter.split_text(docs[0].page_content)
            chunk_texts = chunks
        else:
            chunks = text_splitter.split_documents(docs)
            chunk_texts = [doc.page_content for doc in chunks]
        
        return chunk_texts, len(chunk_texts)
    except Exception as e:
        st.error(f"Error loading book: {e}")
        return [], 0

@st.cache_resource
def setup_model():
    """Setup the ChatOpenAI model (cached for performance)"""
    return ChatOpenAI(model="mistralai/mistral-7b-instruct:free", temperature=0.1)

def simple_retrieval(query, chunk_texts, top_k=3):
    """Simple keyword-based retrieval"""
    query_words = query.lower().split()
    scores = []
    
    for i, chunk in enumerate(chunk_texts):
        chunk_lower = chunk.lower()
        score = sum(1 for word in query_words if word in chunk_lower)
        scores.append((score, i, chunk))
    
    # Sort by score and return top_k
    scores.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, _, chunk in scores[:top_k] if score > 0]

def get_answer(query, chunk_texts, model):
    """Get answer using RAG"""
    # Create the RAG prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context from a book.
        Use only the information in the context to answer questions. If the context doesn't contain 
        relevant information, say so politely.
if user types Hi, Hello or any other greeting, return a simple greeting and introduce yourself"""),
        ("human", """Context from the book:
{context}

Question: {question}

Please provide a helpful answer based on the context above.""")
    ])
    
    chain = prompt_template | model | StrOutputParser()
    
    # Retrieve relevant chunks
    relevant_chunks = simple_retrieval(query, chunk_texts, top_k=3)
    
    if not relevant_chunks:
        return "I couldn't find relevant information in your book for that question.", []
    
    # Combine chunks as context
    context = "\n\n".join(relevant_chunks)
    
    # Get answer
    answer = chain.invoke({
        "context": context, 
        "question": query
    })
    
    return answer, relevant_chunks

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Book Chatbot 🤖</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Settings")
        
        # File Upload Section
        st.subheader("📁 Upload Your Document")
        
        # Show available file types
        supported_types = ['txt', 'md', 'csv', 'py', 'js', 'html', 'css', 'json']
        if PDF_AVAILABLE:
            supported_types.append('pdf')
        if DOCX_AVAILABLE:
            supported_types.append('docx')
        if EXCEL_AVAILABLE:
            supported_types.extend(['xlsx', 'xls'])
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=supported_types,
            help=f"Upload any supported file: {', '.join(supported_types)}"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            file_details = {
                "filename": uploaded_file.name,
                "filetype": uploaded_file.type,
                "filesize": f"{uploaded_file.size:,} bytes"
            }
            st.json(file_details)
            
            # Show preview of the file (except for binary files)
            file_name = uploaded_file.name.lower()
            if not (file_name.endswith(('.pdf', '.docx', '.xlsx', '.xls'))):
                with st.expander("👀 Preview uploaded content"):
                    try:
                        content = uploaded_file.getvalue()
                        if isinstance(content, bytes):
                            content = content.decode('utf-8')
                        st.text_area("File Preview", content[:500] + "..." if len(content) > 500 else content, height=100)
                    except UnicodeDecodeError:
                        st.error("Could not decode file. Please upload a text file.")
            else:
                st.info("� Binary file uploaded successfully!")
        else:
            st.info("�📖 Using default book.txt")
            st.caption("💡 Upload your own file to chat with different content!")
            
        # Show installation status for optional dependencies
        if not PDF_AVAILABLE or not DOCX_AVAILABLE or not EXCEL_AVAILABLE:
            with st.expander("⚠️ Optional Dependencies"):
                if not PDF_AVAILABLE:
                    st.warning("📄 PDF support not available. Install 'pypdf' to read PDF files.")
                if not DOCX_AVAILABLE:
                    st.warning("📝 DOCX support not available. Install 'python-docx' to read Word files.")
                if not EXCEL_AVAILABLE:
                    st.warning("📊 Excel support not available. Install 'pandas' and 'openpyxl' to read Excel files.")
        
        st.divider()
        
        # API Key check
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ API Key loaded")
            st.text(f"Key: {api_key[:15]}...")
        else:
            st.error("❌ No API Key found")
            st.info("Make sure your .env file contains OPENAI_API_KEY")
        
        # Model info
        st.info("🤖 Using: mistralai/mistral-7b-instruct:free")
        
        # Book stats
        sample_content = st.session_state.get('sample_content', None)
        chunk_texts, num_chunks = load_book_data(uploaded_file, sample_content)
        st.metric("📖 Text Chunks", num_chunks)
        
        if num_chunks > 0:
            total_chars = sum(len(chunk) for chunk in chunk_texts)
            st.metric("📄 Total Characters", f"{total_chars:,}")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Reset book cache button (useful when switching files)
        if st.button("🔄 Reload Book"):
            st.cache_data.clear()
            st.rerun()
        
        # Sample files section
        st.subheader("📚 Try Sample Content")
        sample_texts = {
            "Python Tutorial": """
# Python Basics

Python is a high-level programming language known for its simplicity and readability. 

## Variables
In Python, you can create variables easily:
```python
name = "Alice"
age = 25
```

## Functions
Functions are defined using the def keyword:
```python
def greet(name):
    return f"Hello, {name}!"
```

## Data Structures
Python has several built-in data structures:
- Lists: [1, 2, 3]
- Dictionaries: {"key": "value"}
- Tuples: (1, 2, 3)
""",
            "Machine Learning Basics": """
# Machine Learning Introduction

Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data.

## Types of Machine Learning

### Supervised Learning
Uses labeled data to train models. Examples:
- Classification: Predicting categories
- Regression: Predicting continuous values

### Unsupervised Learning
Finds patterns in data without labels. Examples:
- Clustering: Grouping similar data
- Dimensionality Reduction: Simplifying data

### Reinforcement Learning
Learns through interaction with environment using rewards and penalties.

## Popular Algorithms
- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines
""",
            "Web Development Guide": """
# Web Development Fundamentals

Web development involves creating websites and web applications.

## Frontend Technologies
- HTML: Structure of web pages
- CSS: Styling and layout
- JavaScript: Interactivity and behavior

## Backend Technologies
- Python: Django, Flask
- JavaScript: Node.js, Express
- Java: Spring Boot
- PHP: Laravel

## Databases
- SQL: MySQL, PostgreSQL
- NoSQL: MongoDB, Firebase

## Development Process
1. Planning and Design
2. Frontend Development
3. Backend Development
4. Testing
5. Deployment
"""
        }
        
        selected_sample = st.selectbox(
            "Choose a sample to explore:",
            ["None"] + list(sample_texts.keys())
        )
        
        if selected_sample != "None":
            if st.button(f"📖 Load {selected_sample}"):
                # Store sample content in session state
                st.session_state.sample_content = sample_texts[selected_sample]
                st.success(f"✅ Loaded {selected_sample}!")
                st.cache_data.clear()  # Clear cache to reload with new content
                st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Check if we have chunks to work with
    if num_chunks == 0:
        st.warning("⚠️ No book content loaded. Please upload a file or check your book.txt file.")
        st.info("💡 Tip: Upload a .txt file using the sidebar to get started!")
        return
    
    # Display chat messages
    for message_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander(f"📚 View {len(message['sources'])} source(s)"):
                    for i, source in enumerate(message["sources"], 1):
                        st.text_area(
                            f"Source {i}", 
                            source[:300] + "..." if len(source) > 300 else source,
                            height=100,
                            key=f"history_msg_{message_idx}_source_{i}"
                        )
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your book..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Setup model
                    model = setup_model()
                    
                    # Get current book data (in case file was changed)
                    sample_content = st.session_state.get('sample_content', None)
                    current_chunks, _ = load_book_data(uploaded_file, sample_content)
                    
                    # Get answer
                    answer, sources = get_answer(prompt, current_chunks, model)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander(f"📚 View {len(sources)} source(s)"):
                            for i, source in enumerate(sources, 1):
                                st.text_area(
                                    f"Source {i}", 
                                    source[:300] + "..." if len(source) > 300 else source,
                                    height=100,
                                    key=f"current_response_source_{i}_{len(st.session_state.messages)}"
                                )
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
