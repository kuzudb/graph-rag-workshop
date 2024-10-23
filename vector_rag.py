import os

import lancedb
from dotenv import load_dotenv
from ell import ell
from openai import OpenAI


class VectorRAG:
    def __init__(self, db_path: str, table_name: str = "vectors"):
        load_dotenv()
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)

    def query(self, query_vector: list, limit: int = 10) -> list:
        search_result = (
            self.table.search(query_vector).metric("cosine").select(["text"]).limit(limit)
        ).to_list()
        return search_result if search_result else None

    def embed(self, query: str) -> list:
        # For now just using an OpenAI embedding model
        response = self.openai_client.embeddings.create(model="text-embedding-3-small", input=query)
        return response.data[0].embedding

    @ell.simple(model="gpt-4o-mini", temperature=0.3)
    def retrieve(self, query: str, context: str) -> str:
        """
        You are an AI assistant using Retrieval-Augmented Generation (RAG).
        RAG enhances your responses by retrieving relevant information from a knowledge base.
        You will be provided with a query and relevant context. Use only this context to answer the question.
        Do not make up an answer. If you don't know the answer, say so clearly.
        Always strive to provide concise, helpful, and context-aware answers.
        """

        return f"""
        Given the following query and relevant context, please provide a comprehensive and accurate response:

        Query: {query}

        Relevant context:
        {context}

        Response:
        """

    def run(self, question: str) -> str:
        question_embedding = self.embed(question)
        context = self.query(question_embedding)
        return self.retrieve(question, context)


if __name__ == "__main__":
    vector_rag = VectorRAG("./test_lancedb")
    question = "Who are the founders of BlackRock? Return the names as a numbered list."
    response = vector_rag.run(question)
    print(response)

    question = "Where did Larry Fink graduate from?"
    response = vector_rag.run(question)
    print(response)

    question = "When were Larry Fink and Susan Wagner born?"
    response = vector_rag.run(question)
    print(response)
