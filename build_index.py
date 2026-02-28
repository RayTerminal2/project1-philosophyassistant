# build_index.py - 图书管理员脚本
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import chromadb

# --- 1. 配置模型 ---
print("正在初始化模型配置...")

# 设置嵌入模型 (Embedding Model)
# 这是一个专门把中文转化成向量的免费模型，第一次运行会自动下载，大概 300MB-500MB
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 我们在建索引的时候不需要 LLM (DeepSeek) 参与生成，只需要 Embedding
# 所以这里设为 None，省钱，也防止报错
Settings.llm = None 

# --- 2. 设置数据库路径 ---
# 让数据库存在刚才建好的 chroma_db 文件夹里
db_path = "./chroma_db"
if not os.path.exists(db_path):
    os.makedirs(db_path)

print(f"数据库将存储在: {db_path}")

# --- 3. 读取 PDF ---
print("正在读取 data 文件夹中的书籍...")
# 这一步会自动把 data 目录下所有的 pdf, txt 读进来
documents = SimpleDirectoryReader("./data").load_data()
print(f"成功读取了 {len(documents)} 页文档。")

# --- 4. 建立索引并存储 ---
print("正在将文字切片并存入数据库（这可能需要一点时间）...")

# 初始化 Chroma 客户端
chroma_client = chromadb.PersistentClient(path=db_path)
chroma_collection = chroma_client.get_or_create_collection("philosophy_collection")

# 设置存储上下文
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 开始干活：切片 -> 向量化 -> 存入硬盘
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context,
    show_progress=True # 显示进度条
)

print("🎉 恭喜！索引构建完成！")
print("你的书已经被'消化'并存入了 chroma_db 文件夹。")