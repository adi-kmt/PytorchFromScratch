# From twitter.com/jxnlco

from typing import List, Optional
from kg import KnowledgeGraph


def generate_graph(inputs: List[str]) -> KnowledgeGraph:
	curr_state = KnowledgeGraph()

	num_iterations = len(inputs)

	for i, inp in enumerate(inputs):
		update = ""  # Pass llm input here

		curr_state = curr_state.update(update)

	return curr_state


text_chunks = ["", "", ""]

graph: KnowledgeGraph = generate_graph(text_chunks)
