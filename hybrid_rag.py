import os

import cohere
from dotenv import load_dotenv
from ell import ell
from openai import OpenAI

from graph_rag import GraphRAG
from vector_rag import VectorRAG

load_dotenv()
MODEL_NAME = "gpt-4o-mini"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class HybridRAG:
    def __init__(self, graph_db_path="./test_kuzudb", vector_db_path="./test_lancedb"):
        self.graph_rag = GraphRAG(graph_db_path)
        self.vector_rag = VectorRAG(vector_db_path)
        self.co = cohere.ClientV2(COHERE_API_KEY)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

    @ell.simple(model=MODEL_NAME, temperature=0.3)
    def hybrid_rag(self, question: str, context: str) -> str:
        """
        You are an AI assistant using Retrieval-Augmented Generation (RAG).
        RAG enhances your responses by retrieving relevant information from a knowledge base.
        You will be provided with a question and relevant context. Use only this context to answer the question.
        Do not make up an answer. If you don't know the answer, say so clearly.
        Always strive to provide concise, helpful, and context-aware answers.
        """

        return f"""
        Given the following question and relevant context, please provide a comprehensive and accurate response:

        Question: {question}

        Relevant context:
        {context}

        Response:
        """

    def run(self, question: str) -> str:
        question_embedding = self.vector_rag.embed(question)
        vector_docs = self.vector_rag.query(question_embedding)
        vector_docs = [doc["text"] for doc in vector_docs]

        cypher = self.graph_rag.generate_cypher(question)
        graph_docs = self.graph_rag.query(question, cypher)

        docs = [graph_docs] + vector_docs
        # Ensure the doc contents are strings
        docs = [str(doc) for doc in docs]

        combined_context = self.co.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=docs,
            top_n=5,
            return_documents=True,
        )
        return self.hybrid_rag(question, combined_context)


if __name__ == "__main__":
    hybrid_rag = HybridRAG()
    question = "Who are the founders of BlackRock? Return the names as a numbered list."
    response = hybrid_rag.run(question)
    print(response)

    question = "Where did Larry Fink graduate from?"
    response = hybrid_rag.run(question)
    print(response)

    question = "When were Larry Fink and Susan Wagner born?"
    response = hybrid_rag.run(question)
    print(response)
