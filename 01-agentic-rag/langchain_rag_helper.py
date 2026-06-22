from dataclasses import dataclass
from typing import Any

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


AGENT_INSTRUCTIONS = """
You're a course teaching assistant.

Answer the student's question using the search tool.
Make multiple searches with different keywords before answering.

Use only information found through the search tool.
If the answer is not found, say: "I don't know."
"""


@dataclass
class AgenticRAGResponse:
    answer: str
    search_call_count: int
    search_queries: list[str]
    raw_result: Any


class LangChainAgenticRAG:
    def __init__(
        self,
        index,
        model="gpt-5-mini",
        num_results=5,
        instructions=AGENT_INSTRUCTIONS,
        temperature=0,
    ):
        self.index = index
        self.model = model
        self.num_results = num_results
        self.instructions = instructions
        self.temperature = temperature

        self.search_tool = self._build_search_tool()

        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[self.search_tool],
            system_prompt=self.instructions,
        )

    def _format_search_results(self, results):
        lines = []

        for i, doc in enumerate(results, start=1):
            lines.append(f"RESULT {i}")
            if "filename" in doc:
                lines.append(f"filename: {doc['filename']}")
            if "content" in doc:
                lines.append(f"content: {doc['content']}")
            lines.append("")

        return "\n".join(lines).strip()

    def _build_search_tool(self):
        index = self.index
        num_results = self.num_results
        formatter = self._format_search_results

        @tool
        def search(query: str) -> str:
            """
            Search the course lesson index for relevant information.

            Args:
                query: Search keywords or a question to look up in the course material.

            Returns:
                Relevant course chunks as plain text.
            """

            results = index.search(
                query,
                num_results=num_results,
            )

            return formatter(results)

        return search

    def count_tool_calls(self, result, tool_name="search"):
        count = 0

        for message in result["messages"]:
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                continue

            for tool_call in tool_calls:
                if tool_call.get("name") == tool_name:
                    count += 1

        return count

    def extract_search_queries(self, result, tool_name="search"):
        queries = []

        for message in result["messages"]:
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                continue

            for tool_call in tool_calls:
                if tool_call.get("name") == tool_name:
                    args = tool_call.get("args", {})
                    queries.append(args.get("query"))

        return queries

    def rag(self, question):
        result = self.agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        })

        answer = result["messages"][-1].content
        search_call_count = self.count_tool_calls(result)
        search_queries = self.extract_search_queries(result)

        return AgenticRAGResponse(
            answer=answer,
            search_call_count=search_call_count,
            search_queries=search_queries,
            raw_result=result,
        )