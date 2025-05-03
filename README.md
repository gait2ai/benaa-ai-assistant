# Benaa Assistant (AI Chatbot)

Benaa Assistant is an AI-powered chatbot designed to provide users with accurate, friendly, and responsive answers based on the internal documents of **Benaa Association for Human Development (BCFHD)**. It uses Retrieval-Augmented Generation (RAG) techniques, combining document embeddings with OpenAI’s GPT API to deliver informed and contextual responses.

## Features

- **Multi-language support**: Automatically detects the user’s language and responds accordingly.
- **Smart filtering**: Prioritizes relevant file types such as DOCX, PDF, and Markdown.
- **Custom instructions**: Uses a configurable system prompt to control the assistant's behavior.
- **FastAPI backend**: Lightweight and fast API service for deploying the assistant.
- **Interactive frontend**: Simple and responsive web interface using HTML, CSS, and JavaScript.
- **Open-source ready**: Easily customizable and deployable on platforms like GitHub + Railway.

## Folder Structure

```
├── main.py                 # FastAPI backend server
├── embed_documents.py     # Script to generate document embeddings
├── requirements.txt        # Python dependencies
├── static/                 # HTML, CSS, JS frontend
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── favicon.ico
├── data/                   # Internal documents for knowledge base
├── embeddings/             # FAISS index and metadata files
├── config/                 # Custom system prompt and configs
│   └── system_prompt.txt
├── LICENSE.md              # Project license
└── README.md               # This file
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/benaa-assistant.git
cd benaa-assistant
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Add your OpenAI key

Create a `.env` file and add:

```
OPENAI_API_KEY=your_openai_key
```

### 4. Add your documents

Place all internal documents in the `data/` folder.

### 5. Generate embeddings

```bash
python embed_documents.py
```

### 6. Run the app

```bash
uvicorn main:app --reload
```

Then open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Deployment

You can deploy this project on platforms like **Railway** or **Render** by connecting your GitHub repository and setting the environment variable `OPENAI_API_KEY`.

## License

See [LICENSE.md](LICENSE.md) for details.
