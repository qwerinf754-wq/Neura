"""
Neura Agent Orchestrator

Pipeline:

User
  -> ToolAgent
  -> WebContextBuilder
  -> Web Search
  -> Web Fetch
  -> HTML Extraction
  -> Citation-aware Prompt
  -> Model
  -> Citation Validation
"""

from .tool_agent import ToolAgent
from .web_context import WebContextBuilder
from .citations import CitationFormatter


class NeuraOrchestrator:

    def __init__(
        self,
        model_callback=None,
        tool_agent=None,
        web_context_builder=None,
        citation_formatter=None
    ):

        self.model_callback = model_callback

        self.tool_agent = (
            tool_agent
            if tool_agent is not None
            else ToolAgent()
        )

        self.web_context_builder = (
            web_context_builder
            if web_context_builder is not None
            else WebContextBuilder()
        )

        self.citation_formatter = (
            citation_formatter
            if citation_formatter is not None
            else CitationFormatter()
        )

    def set_model(self, model_callback):

        self.model_callback = model_callback

    def _model_generate(self, prompt):

        if self.model_callback is None:
            return None

        return self.model_callback(
            prompt
        )

    def _build_web_prompt(
        self,
        user_text,
        web_data
    ):

        documents = web_data.get(
            "documents",
            []
        )

        web_context = (
            self.web_context_builder.format_context(
                web_data
            )
        )

        source_block = (
            self.citation_formatter.build_source_block(
                documents
            )
        )

        citation_instruction = (
            self.citation_formatter.build_instruction()
        )

        return (
            "You are Neura, an AI assistant.\n\n"

            "Answer the user's question using "
            "the provided web context when relevant.\n\n"

            f"{citation_instruction}\n\n"

            "USER QUESTION:\n"
            f"{user_text}\n\n"

            "WEB CONTEXT:\n"
            f"{web_context}\n\n"

            f"{source_block}\n\n"

            "IMPORTANT:\n"
            "Citations must use the format [N].\n"
            "N must correspond to an existing source.\n"
            "Never create a source number that does not exist."
        )

    def run(
        self,
        user_text,
        max_results=5
    ):

        decision = self.tool_agent.decide(
            user_text
        )

        # -------------------------------------------------
        # Normal model path
        # -------------------------------------------------

        if decision["action"] == "none":

            response = self._model_generate(
                user_text
            )

            return {
                "action": "model",
                "query": user_text,
                "results": [],
                "documents": [],
                "web_context": "",
                "response": response,
                "citations": [],
                "citation_validation": {
                    "valid": True,
                    "citations": [],
                    "invalid": []
                }
            }

        # -------------------------------------------------
        # Tool-backed paths
        # -------------------------------------------------

        if decision["action"] in (
            "github",
            "official_site",
            "knowledge_source",
        ):

            web_data = self.tool_agent.run_context(
                user_text,
                max_results=max_results,
            )

            results = web_data.get(
                "results",
                [],
            )

            documents = web_data.get(
                "documents",
                [],
            )

            web_context = (
                self.web_context_builder.format_context(
                    web_data
                )
            )

            prompt = self._build_web_prompt(
                user_text,
                web_data,
            )

            response = self._model_generate(
                prompt
            )

            if response is None:
                response = ""

            citation_validation = (
                self.citation_formatter.validate_citations(
                    response,
                    documents,
                )
            )

            return {
                "action": decision["action"],
                "query": user_text,
                "results": results,
                "documents": documents,
                "web_context": web_context,
                "response": response,
                "citations": citation_validation[
                    "citations"
                ],
                "citation_validation": citation_validation,
            }

        # -------------------------------------------------
        # Wikipedia RAG path
        # -------------------------------------------------

        if decision["action"] == "wikipedia":

            web_data = self.tool_agent.run_context(
                user_text,
                max_results=max_results,
            )

            results = web_data.get(
                "results",
                [],
            )

            documents = web_data.get(
                "documents",
                [],
            )

            web_context = (
                self.web_context_builder.format_context(
                    web_data
                )
            )

            prompt = self._build_web_prompt(
                user_text,
                web_data,
            )

            response = self._model_generate(
                prompt
            )

            if response is None:
                response = ""

            citation_validation = (
                self.citation_formatter.validate_citations(
                    response,
                    documents,
                )
            )

            return {
                "action": "wikipedia",
                "query": user_text,
                "results": results,
                "documents": documents,
                "web_context": web_context,
                "response": response,
                "citations": citation_validation[
                    "citations"
                ],
                "citation_validation": citation_validation,
            }

        # -------------------------------------------------
        # Web path
        # -------------------------------------------------

        if decision["action"] == "web_search":

            web_data = (
                self.web_context_builder.build(
                    user_text
                )
            )

            results = web_data.get(
                "results",
                []
            )

            documents = web_data.get(
                "documents",
                []
            )

            web_context = (
                self.web_context_builder.format_context(
                    web_data
                )
            )

            prompt = self._build_web_prompt(
                user_text,
                web_data
            )

            response = self._model_generate(
                prompt
            )

            if response is None:
                response = ""

            citation_validation = (
                self.citation_formatter.validate_citations(
                    response,
                    documents
                )
            )

            return {
                "action": "web_search",
                "query": user_text,
                "results": results,
                "documents": documents,
                "web_context": web_context,
                "response": response,
                "citations": citation_validation[
                    "citations"
                ],
                "citation_validation": citation_validation
            }

        raise RuntimeError(
            "Unsupported agent action: "
            + str(
                decision["action"]
            )
        )
