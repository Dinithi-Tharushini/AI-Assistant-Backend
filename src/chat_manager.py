from typing import TypedDict, List, Annotated, Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import uuid

class ChatState(TypedDict, total=False):
    """State for the chat agent with LangGraph add_messages aggregation."""
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[str]
    persist_only: bool

class ChatManager:
    def __init__(self, vector_store):
        """
        Initialize chat manager with LangGraph
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(temperature=0.1, streaming=True)
        # In-memory checkpointer as per LangGraph docs
        self.memory = InMemorySaver()
        
        # Create the chat prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Advantis Assistant, an AI for Advantis employees. Always:
            - Answer using the provided context only.
            - Reflect Advantis values: honesty, integrity, inclusion, teamwork, innovation, sustainability, and humility.
            - Encourage collaboration, learning, and psychological safety.
            - Avoid condoning harassment, discrimination, or dishonesty.
            - Be professional, supportive, and clear.
            - Guide users to the right resources if unsure, while staying aligned with company priorities and culture.
            - Answer as if you are part of the Advantis team.

            Formatting rules:

            - Use Markdown with clear headings and short paragraphs.
            - Use proper lists: each item starts with '- ' or '1.' on a new line.
            - Keep answers concise and structured.
            - Do not combine list items on a single line.

            If the answer is not in the context, politely say so.

            Context: {context}"""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{question}")
        ])
        
        # Simple LLM chain: prompt -> model
        self.chain = self.prompt | self.llm

        # Build the graph
        self.workflow = self._build_graph()

    def _retrieve_context(self, question: str) -> List[str]:
        """
        Retrieve relevant context from vector store
        """
        results = self.vector_store.similarity_search(question)
        return [text for text, _ in results]

    def _build_graph(self):
        """
        Build the conversation graph
        """
        def retrieve_and_answer(state: ChatState):
            # If we're only persisting provided messages, skip generation
            if state.get("persist_only"):
                return {"context": state.get("context", [])}
            # Determine the latest user question from messages
            question = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    question = msg.content
                    break
            if question is None:
                question = ""

            # Retrieve context for this turn
            context = self._retrieve_context(question)

            # Generate response
            ai_msg: AIMessage = self.chain.invoke({
                "context": "\n".join(context),
                "messages": state["messages"],
                "question": question,
            })

            # Return only the new AI message; add_messages will accumulate
            return {"messages": [ai_msg], "context": context}

        # Create the graph
        graph = StateGraph(ChatState)

        # Add the main conversation node
        graph.add_node("retrieve_and_answer", retrieve_and_answer)

        # Set entry and termination
        graph.set_entry_point("retrieve_and_answer")
        graph.add_edge("retrieve_and_answer", END)

        return graph.compile(checkpointer=self.memory)

    def _config_for_session(self, session_id: str):
        return {"configurable": {"thread_id": session_id}}

    def get_response(self, question: str, session_id=None):
        """
        Get response for a question using LangGraph
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        config = self._config_for_session(session_id)

        # Invoke graph with the new human message; memory/checkpointer persists state by thread_id
        final_state = self.workflow.invoke({
            "messages": [HumanMessage(content=question)]
        }, config)

        # Get last AI message
        last_ai_message = next((
            (m for m in reversed(final_state.get("messages", [])) if isinstance(m, AIMessage))
        ), None)

        answer = last_ai_message.content if last_ai_message else ""
        return {
            "session_id": session_id,
            "answer": answer,
            "context": final_state.get("context", []),
        }

    def get_chat_history(self, session_id):
        """
        Get chat history for a session
        """
        if not session_id:
            return []
        snapshot = self.workflow.get_state(self._config_for_session(session_id))
        values = getattr(snapshot, "values", {}) if snapshot else {}
        messages: List[BaseMessage] = values.get("messages", [])
        history = []
        i = 0
        while i < len(messages):
            if isinstance(messages[i], HumanMessage):
                q = messages[i].content
                a = messages[i + 1].content if i + 1 < len(messages) and isinstance(messages[i + 1], AIMessage) else ""
                history.append({"question": q, "answer": a})
                i += 2
            else:
                i += 1
        return history

    def stream_response(self, question: str, session_id: str | None = None) -> tuple[str, Generator[str, None, None]]:
        """Stream tokens for an answer; persists the turn after streaming completes."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        config = self._config_for_session(session_id)
        # Load prior messages from checkpoint
        snapshot = self.workflow.get_state(config)
        values = getattr(snapshot, "values", {}) if snapshot else {}
        prior_messages: List[BaseMessage] = values.get("messages", [])

        context = self._retrieve_context(question)

        def generator() -> Generator[str, None, None]:
            collected_parts: List[str] = []
            buffer = ""  # Buffer to accumulate subword tokens
            last_emitted_char = ""  # Track last char sent to client

            def normalize_fragment(fragment: str) -> str:
                """Ensure natural spacing even if upstream tokens omit spaces."""
                nonlocal last_emitted_char
                piece = fragment
                if piece:
                    needs_prefix_space = (
                        not piece[0].isspace()
                        and last_emitted_char
                        and not last_emitted_char.isspace()
                        and last_emitted_char not in '({["\''
                        and piece[0].isalnum()
                    )
                    if needs_prefix_space:
                        piece = " " + piece

                    stripped = piece.rstrip("\n\r")
                    if stripped:
                        last_emitted_char = stripped[-1]
                return piece
            
            # Build prompt messages explicitly and stream from the model directly
            formatted_msgs = self.prompt.format_messages(
                context="\n".join(context),
                messages=prior_messages,
                question=question,
            )
            
            for chunk in self.llm.stream(formatted_msgs):
                token = getattr(chunk, "content", "")
                if not token:
                    continue
                
                collected_parts.append(token)
                
                # Token normalization to handle subword splits:
                # Strategy: Buffer tokens until we hit a clear word boundary
                
                # Check if token starts with space/newline - this indicates a NEW word
                if token.startswith((' ', '\n', '\t')):
                    # Flush any buffered subwords first
                    if buffer:
                        yield normalize_fragment(buffer)
                        buffer = ""
                    # Now yield this token (space + word or just space)
                    yield normalize_fragment(token)
                # Check if token is pure punctuation or ends with space/newline
                elif token.strip() in '.,!?;:"\'-—…()[]{}' or token.endswith((' ', '\n', '\t')):
                    # Add to buffer and flush (punctuation ends a word)
                    buffer += token
                    yield normalize_fragment(buffer)
                    buffer = ""
                else:
                    # This is a subword fragment (like "Ch", "ann", "aka")
                    # Keep accumulating until we hit a boundary
                    buffer += token
            
            # Flush any remaining buffer at the end of stream
            if buffer:
                yield normalize_fragment(buffer)

            # Persist the turn (user + final ai) to memory for this thread
            try:
                # Persist messages only; avoid re-generating by passing persist_only=True
                self.workflow.invoke({
                    "messages": [HumanMessage(content=question), AIMessage(content="".join(collected_parts))],
                    "context": context,
                    "persist_only": True,
                }, config)
            except Exception:
                # do not break streaming flow if persistence fails
                pass

        return session_id, generator()