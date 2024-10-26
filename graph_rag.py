import os

import kuzu
from dotenv import load_dotenv
from ell import ell
from openai import OpenAI

import prompts

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
SEED = 42


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

        # Handle both hashable and non-hashable types
        if all(isinstance(x, (str, int, float, bool, tuple)) for x in result):
            final_result = {question: list(set(result))}
        else:
            # For non-hashable types, we can't use set() directly
            # Instead, we'll use a list comprehension to remove duplicates
            final_result = {question: [x for i, x in enumerate(result) if x not in result[:i]]}

        return final_result

    @ell.simple(model=MODEL_NAME, temperature=0.1, client=OpenAI(api_key=OPENAI_API_KEY), seed=SEED)
    def generate_cypher(self, question: str) -> str:
        return [
            ell.system(prompts.CYPHER_SYSTEM_PROMPT),
            ell.user(
                prompts.CYPHER_USER_PROMPT.format(schema=self.get_schema(), question=question)
            ),
        ]

    @ell.simple(model=MODEL_NAME, temperature=0.3, client=OpenAI(api_key=OPENAI_API_KEY), seed=SEED)
    def retrieve(self, question: str, context: str) -> str:
        return [
            ell.system(prompts.RAG_SYSTEM_PROMPT),
            ell.user(prompts.RAG_USER_PROMPT.format(question=question, context=context)),
        ]

    def run(self, question: str) -> str:
        cypher = self.generate_cypher(question)
        print(f"\n{cypher}\n")
        context = self.query(question, cypher)
        return self.retrieve(question, context)


if __name__ == "__main__":
    graph_rag = GraphRAG("./test_kuzudb")
    question = "Who are the founders of BlackRock? Return the names as a numbered list."
    response = graph_rag.run(question)
    print(f"Q1: {question}\n\n{response}\n---\n")

    question = "Where did Larry Fink graduate from?"
    response = graph_rag.run(question)
    print(f"Q2: {question}\n\n{response}\n---\n")

    question = "When was Susan Wagner born?"
    response = graph_rag.run(question)
    print(f"Q3: {question}\n\n{response}\n---\n")

    question = "How did Larry Fink and Rob Kapito meet?"
    response = graph_rag.run(question)
    print(f"---\nQ4: {question}\n\n{response}")
