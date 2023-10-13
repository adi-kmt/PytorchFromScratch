from typing import List, Optional
from pydantic import BaseModel, Field


class Node(BaseModel):
	id: int
	label: str
	colour: str

	def __hash__(self) -> int:
		return hash(self.label)


class Edge(BaseModel):
	source: int
	target: int
	label: str
	colour: str = "black"

	def __hash__(self):
		return hash((self.source, self.target, self.label))


class KnowledgeGraph(BaseModel):
	nodes: Optional[List[Node]] = Field(..., default_factory=list)
	edges: Optional[List[Edge]] = Field(..., default_factory=list)

	def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
		return KnowledgeGraph(
			nodes=list(set(self.nodes + other.nodes)),
			edges=list(set(self.edges + other.edges))
		)
