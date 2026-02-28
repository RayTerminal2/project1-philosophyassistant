import streamlit as st
from openai import OpenAI
import os
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, Settings

# --- 1. 页面基础设置 ---
st.set_page_config(page_title="西哲考研助手", page_icon="🦉", layout="wide")
MY_API_KEY = "sk-4f8a32d787074897b9c9b8de39ecb3a1"
st.title("🏛️ 西方哲学史·对话伴侣 (RAG增强版)")


# --- 2. 加载数据库 (核心黑科技) ---
# 使用 @st.cache_resource 装饰器，保证模型只加载一次，不用每次刷新都重跑
@st.cache_resource
def load_knowledge_base():
    try:
        print("正在加载嵌入模型和数据库...")
        # 1. 必须使用和建库时一模一样的嵌入模型
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
        Settings.embed_model = embed_model
        
        # 2. 连接到本地的 chroma_db
        db_path = "./chroma_db"
        if not os.path.exists(db_path):
            return None # 如果没建库，就返回空
            
        chroma_client = chromadb.PersistentClient(path=db_path)
        chroma_collection = chroma_client.get_collection("philosophy_collection")
        
        # 3. 加载索引
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
        return index
    except Exception as e:
        print(f"加载数据库失败: {e}")
        return None

# 执行加载
with st.spinner("正在启动知识引擎，请稍候..."):
    index = load_knowledge_base()

# --- 3. 侧边栏设置 ---
with st.sidebar:
    st.header("⚙️ 设置")
    
    # 【修改点】不再显示输入框，而是显示一个已连接的状态
    st.success(f"API Key 已配置 (DeepSeek)")
    
    st.markdown("---")
    # 显示知识库状态
    if index:
        st.success("✅ 知识库已连接 (百万级数据索引)")
    else:
        st.warning("⚠️ 未检测到知识库，请先运行 build_index.py")

    st.markdown("---")
    st.header("🧠 模式选择")
    
    role_options = ["📚 考研学术助手 (通用)", "苏格拉底", "柏拉图", "亚里士多德", "康德", "尼采", "维特根斯坦"]
    selected_role = st.selectbox("选择你的对话对象", role_options)
    st.markdown("---")
    st.header("请请神上身")
    philosopher = st.selectbox(
        "选择导师",
        ["苏格拉底", "柏拉图", "亚里士多德", "康德", "尼采", "维特根斯坦"]
    )

    # 简化的 Prompt，我们会动态插入知识
    base_personas = {
        "📚 考研学术助手 (通用)": """
            你是一名专业的哲学考研辅导员。
            你的语言风格应该是：客观、精准、条理清晰。
            不需要任何角色扮演的废话。
            
            回答逻辑：
            1. 直接给出核心定义或答案。
            2. 如果涉及复杂概念，分点论述（1... 2... 3...）。
            3. 总是尝试总结该知识点在哲学史上的地位或影响（这对考研很重要）。
        """,
        
        "苏格拉底": """
            你现在是苏格拉底。
            你的教学目标是：帮助用户通过独立思考理解概念，而不是死记硬背。
            
            行为准则：
            1. 即使检索到了答案，也不要直接全盘托出。
            2. 尝试用一个反问句开始，引导用户自己说出书中的观点。
            3. 语气谦虚，经常说'我只知道我一无所知'。
        """,
        
        "柏拉图": "你现在是柏拉图。回答时请侧重于区分'感官世界'与'理念世界'，强调灵魂的回忆说。",
        
        "亚里士多德": "你现在是亚里士多德。请用百科全书式的风格回答，非常注重'定义'、'范畴'和'逻辑推导'。批评你老师柏拉图的观点。",
        
        "康德": "你现在是康德。请使用严谨的德式学术口吻。重点关注'先天综合判断'、'现象与物自体'的区分。",
        
        "尼采": "你现在是尼采。请用格言警句式的语言，充满激情和批判性，痛斥传统道德的虚伪。"
    }

# --- 4. 聊天记录 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，我是你的哲学导师。我已经阅读了你的参考书，随时准备回答。"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 5. 核心：检索 + 生成 ---
if user_input := st.chat_input("输入问题..."):
    

    # 显示用户提问
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        try:
            # Step A: 去数据库里找资料 (Retrieval)
            retrieved_text = ""
            if index:
                # 建立检索器，每次找最相关的 3 段内容
                retriever = index.as_retriever(similarity_top_k=3)
                nodes = retriever.retrieve(user_input)
                
                # 把找到的内容拼起来
                retrieved_text = "\n\n".join([node.get_text() for node in nodes])
                
                # (可选) 悄悄在折叠框里显示AI查到了什么，方便你核对
                with st.expander("🔍 AI 查阅到的原文片段"):
                    st.text(retrieved_text)

            # Step B: 组装 Prompt (Generation)
            client = OpenAI(api_key=MY_API_KEY, base_url="https://api.deepseek.com")
            
            system_prompt = base_personas.get(selected_role, base_personas["📚 考研学术助手 (通用)"])
            
            if retrieved_text:
                system_prompt += f"""
                \n\n【重要指令】
                请基于以下参考资料回答，并保持你的【{selected_role}】设定（如果有的话）。
                
                【参考书片段】：
                {retrieved_text}
                """
            
            messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages

            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages_to_send,
                stream=True
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"出错: {e}")