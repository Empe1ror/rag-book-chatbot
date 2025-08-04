# RAG Book Chatbot 📚🤖

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot that allows you to upload documents and ask questions about their content.

## Features

- 📄 **Multiple File Formats**: Support for PDF, DOCX, Excel, and text files
- 🤖 **AI-Powered**: Uses OpenAI-compatible models via OpenRouter
- 💬 **Chat Interface**: Interactive conversation with your documents
- 📚 **Source References**: See which parts of your document the answers came from
- 🔄 **Real-time Processing**: Upload and chat with documents instantly

## Supported File Types

- **Text Files**: `.txt`, `.md`, `.csv`, `.py`, `.js`, `.html`, `.css`, `.json`
- **Documents**: `.pdf`, `.docx`
- **Spreadsheets**: `.xlsx`, `.xls`

## Demo

Try the live app: [Your Streamlit App URL]

## Installation

### Prerequisites

- Python 3.8+
- OpenRouter API key (or OpenAI API key)

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/rag-book-chatbot.git
   cd rag-book-chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openrouter_api_key_here
   OPENAI_API_BASE=https://openrouter.ai/api/v1
   ```

4. **Run the app**:
   ```bash
   streamlit run streamlit_rag.py
   ```

## Usage

1. **Upload a Document**: Use the sidebar to upload any supported file
2. **Start Chatting**: Ask questions about your document in the chat interface
3. **View Sources**: Expand the source sections to see where answers came from
4. **Try Samples**: Use built-in sample content to test the app

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenRouter or OpenAI API key
- `OPENAI_API_BASE`: API base URL (default: https://openrouter.ai/api/v1)

### Model Configuration

The app uses `mistralai/mistral-7b-instruct:free` by default. You can modify this in `streamlit_rag.py`:

```python
model = ChatOpenAI(model="your-preferred-model", temperature=0.1)
```

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your environment variables in the Streamlit Cloud settings
5. Deploy!

### Other Platforms

The app can also be deployed on:
- Heroku
- Railway
- Render
- Vercel
- Google Cloud Platform
- AWS

## Project Structure

```
rag-book-chatbot/
├── streamlit_rag.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore              # Git ignore file
├── README.md               # This file
└── book.txt                # Default sample book (optional)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://langchain.com/)
- Uses [OpenRouter](https://openrouter.ai/) for AI model access

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/yourusername/rag-book-chatbot/issues) on GitHub.
