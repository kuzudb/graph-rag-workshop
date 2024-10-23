# Hybrid RAG

Code for the Google Developer Group Surrey's DevFest 2024 workshop on Graph RAG and Hybrid RAG.

The following stack is used:

- Graph database: Kùzu
- Vector database: LanceDB
- LLM prompting framework: [ell](https://docs.ell.so/), a language model prompting framework + OpenAI GPT-4o-mini
- Entity & relationship extraction: LlamaIndex + OpenAI GPT-4o-mini
- Embedding model: OpenAI text-embedding-3-small
- Generation model: OpenAI GPT-4o-mini
- Reranking: Cohere reranker

## Setup

We will be using the Python API of Kùzu and a combination of scripts that utilize the required
dependencies. You can manage dependencies using
`requirements.txt` files, installed via `pip`, or the `uv` package manager (recommended).
