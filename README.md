# GDG Surrey DevFest 2024 Workshop

Code for the Google Developer Group Surrey's DevFest 2024 workshop on Graph RAG and Hybrid RAG.

The following stack is used:

- Graph database: [Kùzu](https://kuzudb.com/)
- Vector database: [LanceDB](https://lancedb.com/)
- LLM prompting framework: [ell](https://docs.ell.so/), a language model prompting framework + OpenAI GPT-4o-mini
- Entity & relationship extraction: [LlamaIndex](https://docs.llamaindex.ai/) + OpenAI GPT-4o-mini
- Embedding model: OpenAI text-embedding-3-small
- Generation model: OpenAI GPT-4o-mini
- Reranking: Cohere [reranker](https://docs.cohere.com/v2/reference/rerank)

## Dataset

The dataset used in this workshop is the [BlackRock founders dataset](./data/blackrock), which
are three small text files containing information about the founders of the asset management firm
BlackRock.

The aim of the workshop is to show how we can build a hybrid RAG system that utilizes a graph
database and a vector database to answer questions about the dataset.

## Setup

We will be using the Python API of Kùzu and a combination of scripts that utilize the required
dependencies.

If using Astral's [`uv` package manager](https://docs.astral.sh/uv/), it's recommended to use a
version of Python managed by `uv` itself. You can install the required version of Python (3.12)
with the following command:

```bash
uv python install 3.12
```

All the dependencies are indicated in the `pyproject.toml` file and the associated `uv.lock` file
provided in this repo. Simply sync the dependencies to your local virtual environment with the
following command:

```bash
uv sync
```

Alternatively you can use your system's Python installation and pip to install the dependencies
via `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Steps

### Construct the graph

The script `construct_graph.py` extracts entities and relationships from the provided
[BlackRock founders dataset](./data/blackrock) and constructs a graph that is stored in Kùzu.

```bash
python construct_graph.py
```

The script `construct_graph.py` does the following:
- Chunk the text, generate embeddings, and stores the embeddings in a [LanceDB](https://lancedb.com/),
an embedded vector database
- Use the LlamaIndex framework and its
[property graph index](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/)
to extract entities and relationships from the unstructured text.
- Store the extracted entities and relationships in Kùzu, an embedded graph database
- Augment the graph with additional entities and relationships obtained from external sources

### Run traditional RAG (vector search) queries

The script `vector_rag.py` runs retrieval-augmented generation (RAG) that leverages semantic
(vector) search. To retrieve from the vector database, the script first embeds the question and then
searches for the nearest neighbors using cosine similarity. It then retrieves the context (chunks of
text) that are most similar to the question. The script finally uses the LLM to generate a response
using the retrieved context.

```bash
python vector_rag.py
```

### Run Graph RAG queries

The script `graph_rag.py` runs retrieval-augmented generation (RAG) that leverages the graph
database to answer questions. To retrieve from the graph database, the script first translates
the question into a Cypher query, which is then executed against the graph database. The retrieved
entities and relationships are then used as context to generate a response using the LLM.

```bash
python graph_rag.py
```

### Run Hybrid RAG queries

The script `hybrid_rag.py` runs retrieval-augmented generation (RAG) that leverages *both* the
vector database and the graph database. The vector and graph retrieval contexts are concatenated
together and passed to the LLM to generate a response.

```bash
python hybrid_rag.py
```
