import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import ChatOpenAI


def get_ai_response_for_document_analysis(user_prompt, model_version, temperature, frequency_penalty, presence_penalty, document_content):
    """专门用于文档分析的AI响应获取函数"""
    try:
        load_dotenv()
        model = ChatOpenAI(
            model=model_version,
            api_key='hk-qs8d101000055444378649f712c688d8f09a39faa8151aa3',
            base_url="https://api.openai-hk.com/v1"
        )
        # 构造包含文档内容的完整提示
        full_prompt = f"用户的问题是：{user_prompt}\n\n文档内容：{document_content}\n\n请根据文档内容回答用户的问题。"
        chain = ConversationChain(llm=model, memory=st.session_state.memory)
        return chain.invoke({'input': full_prompt})['response']
    except Exception as err:
        st.error(f"发生错误：{err}")
        st.write("链接解析失败，可能是由于网络问题或链接本身的合法性问题导致的。请检查链接的合法性，并确保网络连接正常。如果问题仍然存在，建议稍后再试。")
        return '暂时无法获取服务器响应……'
def document_analysis():
    """文档分析功能"""
    st.title("文档分析智能体")

    # 初始化会话状态
    if 'document_content' not in st.session_state:
        st.session_state.document_content = ""


    # 文档上传和读取
    uploaded_file = st.file_uploader("上传你的文档", type=["pdf", "txt"])
    if uploaded_file:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            file_type = uploaded_file.name.split('.')[-1]
            if file_type == "txt":
                loader = TextLoader(tmp_file_path)
                docs = loader.load()
                st.session_state.document_content = docs[0].page_content
            elif file_type == "pdf":
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load_and_split()
                full_text = ""
                for doc in docs:
                    full_text += doc.page_content + "\n"
                st.session_state.document_content = full_text
            st.success("文档读取成功！")
        except Exception as e:
            st.error(f"读取文档时出错：{e}")
        finally:
            # 删除临时文件
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    else:
        # 如果用户取消了文档上传，则清除文档内容
        st.session_state.document_content = ""

    # 显示文档内容
    if st.session_state.document_content:
        with st.expander("查看/折叠文档内容"):
            st.subheader("文档内容")
            st.write(st.session_state.document_content)

    # 侧边栏配置
    with st.sidebar:
        with st.expander("模型配置"):
            model_version = st.selectbox("选择模型版本", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
            temperature = st.slider("设置模型温度", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
            frequency_penalty = st.slider("设置频率惩罚", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            presence_penalty = st.slider("设置存在惩罚", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # 用户输入区域
    user_query = st.text_input("请输入你的问题：")
    generate_response = st.button("生成回答")

    if generate_response and user_query and st.session_state.document_content:
        with st.spinner('AI正在思考，请等待……'):
            resp_from_ai = get_ai_response_for_document_analysis(user_query, model_version, temperature, frequency_penalty, presence_penalty, st.session_state.document_content)
            st.subheader("AI的回答")
            st.write(resp_from_ai)
