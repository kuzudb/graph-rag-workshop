import os

import lancedb
from dotenv import load_dotenv
from ell import ell
from openai import OpenAI

import prompts

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
SEED = 42


class VectorRAG:
    def __init__(self, db_path: str, table_name: str = "vectors"):
        load_dotenv()
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
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

    @ell.simple(model=MODEL_NAME, temperature=0.3, client=OpenAI(api_key=OPENAI_API_KEY), seed=SEED)
    def retrieve(self, question: str, context: str) -> str:
        return [
            ell.system(prompts.RAG_SYSTEM_PROMPT),
            ell.user(prompts.RAG_USER_PROMPT.format(question=question, context=context)),
        ]

    def run(self, question: str) -> str:
        question_embedding = self.embed(question)
        context = self.query(question_embedding)
        return self.retrieve(question, context)


if __name__ == "__main__":
    vector_rag = VectorRAG("./test_lancedb")
    question = "Who are the founders of BlackRock? Return the names as a numbered list."
    response = vector_rag.run(question)
    print(f"Q1: {question}\n\n{response}")

    question = "Where did Larry Fink graduate from?"
    response = vector_rag.run(question)
    print(f"---\nQ2: {question}\n\n{response}")

    question = "When was Susan Wagner born?"
    response = vector_rag.run(question)
    print(f"---\nQ3: {question}\n\n{response}")

    question = "How did Larry Fink and Rob Kapito meet?"
    response = vector_rag.run(question)
    print(f"---\nQ4: {question}\n\n{response}")
