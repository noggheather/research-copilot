###### 📚 Research Copilot

A local-first research document retrieval and RAG assistant powered by FastAPI, Streamlit, and Ollama.

Research Copilot helps users search research documents and generate answers based on retrieved passages, using a locally running LLM without paid API services.



✨ Features

Dual Retrieval: Keyword matching and BM25 ranking.

Local RAG: Generate answers using Qwen3 1.7B via Ollama.

Source Tracking: Display retrieved passages and their source documents.

Web Interface: Interactive Streamlit frontend.

REST API: FastAPI endpoints for search and question answering.

Automated Testing: pytest-based tests for retrieval, API, and RAG logic.



🛠️ Tech Stack

Python · FastAPI · Streamlit · BM25 · Ollama · Qwen3 · pytest



🚀 Quick Start



1\. Clone the repository

git clone https://github.com/noggheather/research-copilot.git

cd research-copilot



2\. Install dependencies

conda create -n research-copilot python=3.11 -y

conda activate research-copilot

pip install -r requirements.txt



3\. Install the local model

Install Ollama, start its service, then run:

ollama pull qwen3:1.7b



4\. Start the backend

python -m uvicorn api:app --reload --app-dir src

API documentation: http://127.0.0.1:8000/docs



5\. Start the frontend in a second terminal:

streamlit run streamlit\_app.py

Open the local URL displayed by Streamlit.



🧪 Tests

Run the automated test suite:

python -m pytest tests -v



🏗️ Architecture



Documents → Chunking → Keyword / BM25 Retrieval

&#x20;                           |

&#x20;                           v

&#x20;                     Retrieved Context

&#x20;                           |

&#x20;                           v

&#x20;                    Local Qwen via Ollama

&#x20;                           |

&#x20;                           v

&#x20;                    Answer + Sources



&#x20;                Streamlit ↔ FastAPI

