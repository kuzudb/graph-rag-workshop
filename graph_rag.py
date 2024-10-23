import os

import kuzu
from dotenv import load_dotenv
from ell import ell
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"


class GraphRAG:
    """Graph Retrieval Augmented Generation from a Kùzu database."""

    def __init__(self, db_path="./test_kuzudb"):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

    def get_schema(self) -> str:
        """Provides the graph schema information for the purposes of Cypher generation via an LLM."""
        node_properties = []
        node_table_names = self.conn._get_node_table_names()
        for table_name in node_table_names:
            current_table_schema = {"properties": [], "label": table_name}
            properties = self.conn._get_node_property_names(table_name)
            for property_name in properties:
                property_type = properties[property_name]["type"]
                list_type_flag = ""
                if properties[property_name]["dimension"] > 0:
                    if "shape" in properties[property_name]:
                        for s in properties[property_name]["shape"]:
                            list_type_flag += "[%s]" % s
                    else:
                        for i in range(properties[property_name]["dimension"]):
                            list_type_flag += "[]"
                property_type += list_type_flag
                current_table_schema["properties"].append((property_name, property_type))
            node_properties.append(current_table_schema)

        relationships = []
        rel_tables = self.conn._get_rel_table_names()
        for table in rel_tables:
            relationships.append("(:%s)-[:%s]->(:%s)" % (table["src"], table["name"], table["dst"]))

        rel_properties = []
        for table in rel_tables:
            table_name = table["name"]
            current_table_schema = {"properties": [], "label": table_name}
            query_result = self.conn.execute(f"CALL table_info('{table_name}') RETURN *;")
            while query_result.has_next():
                row = query_result.get_next()
                prop_name = row[1]
                prop_type = row[2]
                current_table_schema["properties"].append((prop_name, prop_type))
            rel_properties.append(current_table_schema)

        schema = (
            f"Node properties: {node_properties}\n"
            f"Relationships properties: {rel_properties}\n"
            f"Relationships: {relationships}\n"
        )
        return schema

    def query(self, question: str, cypher: str) -> str:
        """Use the generated Cypher statement to query the graph database."""
        response = self.conn.execute(cypher)
        result = []
        while response.has_next():
            item = response.get_next()
            if item not in result:
                result.extend(item)
        return {question: list(set(result))}

    @ell.simple(model=MODEL_NAME, temperature=0.1, client=OpenAI(api_key=OPENAI_API_KEY))
    def generate_cypher(self, question: str) -> str:
        """
        You are an expert in translating natural language questions into Cypher statements.
        You will be provided with a question and a graph schema.
        Use only the provided relationship types and properties in the schema to generate a Cypher statement.
        The Cypher statement could retrieve nodes, relationships, or both.
        Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
        """

        return f"""
        Task:Generate Cypher statement to query a graph database.
        Instructions:
        Schema:
        {self.get_schema()}

        The question is:
        {question}

        Instructions:
        Generate the Kùzu dialect of Cypher with the following rules in mind:
        1. Do not include triple backticks ``` in your response. Return only Cypher.
        2. Only use the nodes and relationships provided in the schema.
        3. Use only the provided node and relationship types and properties in the schema.
        """

    @ell.simple(model=MODEL_NAME, temperature=0.3, client=OpenAI(api_key=OPENAI_API_KEY))
    def retrieve(self, question: str, context: str) -> str:
        """
        You are an AI assistant using Retrieval-Augmented Generation (RAG).
        RAG enhances your responses by retrieving relevant information from a knowledge base.
        You will be provided with a question and relevant context. Use only this context to answer the question.
        Do not make up an answer. If you don't know the answer, say so clearly.
        Always strive to provide concise, helpful, and context-aware answers.
        """

        cypher = self.generate_cypher(question)
        context = self.query(question, cypher)

        return f"""
        Given the following question and relevant context, please provide a comprehensive and accurate response:

        Question: {question}

        Relevant context:
        {context}

        Response:
        """

    def run(self, question: str) -> str:
        cypher = self.generate_cypher(question)
        print(cypher)
        context = self.query(question, cypher)
        print(context)
        return self.retrieve(question, context)


if __name__ == "__main__":
    question = "Who are the founders of BlackRock? Return the names as a numbered list."
    graph_rag = GraphRAG("./test_kuzudb")
    response = graph_rag.run(question)
    print(response)

    question = "Where did Larry Fink graduate from?"
    response = graph_rag.run(question)
    print(response)

    question = "When were Larry Fink and Susan Wagner born?"
    response = graph_rag.run(question)
    print(response)
