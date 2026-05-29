# Langchain-
# 🤖 LangChain Chatbot with Persistent Memory

A conversational AI chatbot built with LangChain and ChromaDB that remembers past conversations and retrieves relevant context using RAG (Retrieval-Augmented Generation).

## Features

- **Persistent Memory** — Stores every conversation turn in ChromaDB; retrieves semantically relevant past exchanges on each new message
- **RAG Support** — Loads external notes/documents into a separate vector store and injects relevant chunks into context
- **Groq LLM** — Powered by `llama-3.1-8b-instant` via the Groq API for fast inference
- **HuggingFace Embeddings** — Uses `all-MiniLM-L6-v2` locally for embedding, no extra API cost
- **Persona** — Configured as Sheldon Cooper (easily customizable via system prompt)

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Env Management | python-dotenv |

## Project Structure

```
├── chatbot.py          # Main chatbot script
├── chat_history.txt    # RAG notes / knowledge base (your custom docs)
├── memory/             # ChromaDB persistent store (conversation history)
├── rag_memory/         # ChromaDB persistent store (RAG notes)
├── .env                # API keys (not committed)
├── .gitignore
└── README.md
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/langchain-memory-chatbot.git
cd langchain-memory-chatbot
```

**2. Install dependencies**
```bash
pip install langchain langchain-groq langchain-chroma langchain-huggingface langchain-community langchain-text-splitters chromadb python-dotenv
```

**3. Set up your `.env` file**
```
GROQ_API_KEY=your_groq_api_key_here
```

**4. Add your RAG notes**

Put any text content you want the bot to reference in `chat_history.txt`. On first run, it auto-loads into the RAG vector store.

**5. Run**
```bash
python chatbot.py
```

Type `exit` to quit.

## How It Works

1. User input is embedded and stored in the **conversation memory** ChromaDB collection
2. Similar past messages are retrieved and injected into context (threshold: cosine score < 0.5)
3. The **RAG notes** store is also queried for relevant document chunks (threshold: score < 1.5)
4. Everything is passed to the LLM along with chat history
5. The AI response is stored back into the memory store

## Configuration

| Variable | Location | Description |
|---|---|---|
| `model` | `chatbot.py` | Swap Groq model (e.g. `llama-3.3-70b-versatile`) |
| `SystemMessage` | `chatbot.py` | Change the bot's persona |
| `chunk_size` | `chatbot.py` | RAG chunk size (default: 500) |
| `k` | `chatbot.py` | Number of memory/RAG results to retrieve |

## Notes

- `memory/` and `rag_memory/` directories are auto-created on first run
- The RAG store only loads documents once — delete `rag_memory/` to force a reload
- Update the `TextLoader` path in `chatbot.py` to point to your notes file
