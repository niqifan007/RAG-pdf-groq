from typing import Optional

from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


def get_groq_assistant(
    llm_model: str = "llama3-70b-8192",
    embeddings_model: str = "text-embedding-3-small",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a RAG Assistant."""

    # 根据嵌入模型定义嵌入器
    embedder = (
        OllamaEmbedder(model=embeddings_model, dimensions=768)
        if embeddings_model == "nomic-embed-text"
        else OpenAIEmbedder(model=embeddings_model, dimensions=1536)
    )
    # 根据嵌入模型定义嵌入表
    embeddings_table = (
        "groq_rag_documents_ollama" if embeddings_model == "nomic-embed-text" else "groq_rag_documents_openai"
    )

    return Assistant(
        name="groq_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=Groq(model=llm_model),
        storage=PgAssistantStorage(table_name="groq_rag_assistant", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=embeddings_table,
                embedder=embedder,
            ),
            # 文档数量引用设置为2
            num_documents=2,
        ),
        description="你是一个名为'Best RAG'的AI和中文智者，你的任务是使用提供的信息回答问题",
        instructions=[
            "当用户提出问题时，你将获得有关该问题的信息。",
            "仔细阅读这些信息，并向用户提供清晰、简洁的答案。",
            "不要使用'根据我的知识'或'取决于信息'等短语。",
            "如果你不确定答案，请告诉用户你不知道。"
        ],
        # 在用户提示中添加来自知识库的引用
        add_references_to_prompt=True,
        # 设置LLM以markdown格式化消息
        markdown=True,
        # 在消息中添加聊天历史
        add_chat_history_to_messages=True,
        # 在消息中添加4条之前的聊天历史
        num_history_messages=4,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )
