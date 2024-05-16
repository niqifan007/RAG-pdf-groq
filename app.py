from typing import List

import streamlit as st
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.utils.log import logger

from assistant import get_groq_assistant  # type: ignore

st.set_page_config(
    page_title="RAG知识问答助手",
    page_icon=":orange_heart:",
)
st.title("RAG with Llama3")
st.markdown("##### :orange_heart: 一个基于RAG的知识问答助手 :orange_heart:")


def restart_assistant():
    st.session_state["rag_assistant"] = None
    st.session_state["rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()


def main() -> None:
    # 获取LLM模型
    llm_model = st.sidebar.selectbox("选择LLM模型", options=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"])
    # 设置会话状态中的助手类型
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # 如果助手类型已更改，则重启助手
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    # 获取嵌入模型
    embeddings_model = st.sidebar.selectbox(
        "选择嵌入模型",
        options=["nomic-embed-text", "text-embedding-3-small"],
        help="更改嵌入模型时，需要重新添加文档。",
    )
    # 设置会话状态中的嵌入模型
    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    # 如果嵌入模型已更改，则重启助手
    elif st.session_state["embeddings_model"] != embeddings_model:
        st.session_state["embeddings_model"] = embeddings_model
        st.session_state["embeddings_model_updated"] = True
        restart_assistant()

    # 获取助手
    rag_assistant: Assistant
    if "rag_assistant" not in st.session_state or st.session_state["rag_assistant"] is None:
        logger.info(f"---*--- 创建 {llm_model} 助手 ---*---")
        rag_assistant = get_groq_assistant(llm_model=llm_model, embeddings_model=embeddings_model)
        st.session_state["rag_assistant"] = rag_assistant
    else:
        rag_assistant = st.session_state["rag_assistant"]

    # 创建助手运行（即记录到数据库）并在会话状态中保存运行ID
    try:
        st.session_state["rag_assistant_run_id"] = rag_assistant.create_run()
    except Exception:
        st.warning("无法创建助手，数据库是否运行？")
        return

    # 加载现有消息
    assistant_chat_history = rag_assistant.memory.get_chat_history()
    if len(assistant_chat_history) > 0:
        logger.debug("加载聊天历史")
        st.session_state["messages"] = assistant_chat_history
    else:
        logger.debug("未找到聊天历史")
        st.session_state["messages"] = [{"role": "assistant", "content": "上传文档并向我提问..."}]

    # Prompt 用户输入
    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})

    # 显示现有聊天消息
    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 如果最后一条消息来自用户，则生成新的响应
    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            for delta in rag_assistant.run(question):
                response += delta  # type: ignore
                resp_container.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

    # 加载知识库
    if rag_assistant.knowledge_base:
        # 添加网站到知识库
        if "url_scrape_key" not in st.session_state:
            st.session_state["url_scrape_key"] = 0

        input_url = st.sidebar.text_input(
            "将URL添加到知识库", type="default", key=st.session_state["url_scrape_key"]
        )
        add_url_button = st.sidebar.button("添加URL")
        if add_url_button:
            if input_url is not None:
                alert = st.sidebar.info("正在处理URL...", icon="ℹ️")
                if f"{input_url}_scraped" not in st.session_state:
                    scraper = WebsiteReader(max_links=2, max_depth=1)
                    web_documents: List[Document] = scraper.read(input_url)
                    if web_documents:
                        rag_assistant.knowledge_base.load_documents(web_documents, upsert=True)
                    else:
                        st.sidebar.error("无法读取网站")
                    st.session_state[f"{input_url}_uploaded"] = True
                alert.empty()

        # 添加PDF到知识库
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 100

        uploaded_file = st.sidebar.file_uploader(
            "添加PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
        )
        if uploaded_file is not None:
            alert = st.sidebar.info("正在处理PDF...", icon="🧠")
            rag_name = uploaded_file.name.split(".")[0]
            if f"{rag_name}_uploaded" not in st.session_state:
                reader = PDFReader()
                rag_documents: List[Document] = reader.read(uploaded_file)
                if rag_documents:
                    rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
                else:
                    st.sidebar.error("无法读取PDF")
                st.session_state[f"{rag_name}_uploaded"] = True
            alert.empty()

    if rag_assistant.knowledge_base and rag_assistant.knowledge_base.vector_db:
        if st.sidebar.button("清空知识库"):
            rag_assistant.knowledge_base.vector_db.clear()
            st.sidebar.success("知识库已清空")

    if rag_assistant.storage:
        rag_assistant_run_ids: List[str] = rag_assistant.storage.get_all_run_ids()
        new_rag_assistant_run_id = st.sidebar.selectbox("运行ID", options=rag_assistant_run_ids)
        if st.session_state["rag_assistant_run_id"] != new_rag_assistant_run_id:
            logger.info(f"---*--- 加载 {llm_model} 运行: {new_rag_assistant_run_id} ---*---")
            st.session_state["rag_assistant"] = get_groq_assistant(
                llm_model=llm_model, embeddings_model=embeddings_model, run_id=new_rag_assistant_run_id
            )
            st.rerun()

    if st.sidebar.button("New Run"):
        restart_assistant()

    if "embeddings_model_updated" in st.session_state:
        st.sidebar.info("由于嵌入模型已更改，请重新添加文档。")
        st.session_state["embeddings_model_updated"] = False


main()
