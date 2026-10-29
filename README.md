##### rag.py:


把检索到的资料交给大模型，让大模型根据资料回答问题，并把答案和来源一起返回。



它主要有四项功能：

\- 整理资料：接收 BM25 找到的文本块，整理成模型能理解的上下文。

\- 组织提问内容：将用户问题、检索资料和回答规则组合成 Prompt。

\- 调用本地模型：通过 HTTP 请求调用 Ollama 中的 Qwen3 1.7B。

\- 整理结果：返回用户问题、生成的答案，以及相关的来源文件和文本片段。

build\_context()：整理检索结果（接收bm25返回的结果，简单的整理格式）



generate\_answer()：调用本地模型



def generate\_answer(query: str, context: str) -> str:

接收两个参数：

\- query：用户提出的问题。

\- context：刚刚整理好的检索上下文。

主要完成两个问题：

一，构造prompt：要求模型尽量依据检索到的资料回答，并在证据不足时说明情况。

prompt = f"""

You are a research assistant.

Answer the question using only the provided context.

If the context does not contain enough information, say so.

Do not invent facts.



Context:

{context}



Question:

{query}



Answer in the same language as the question.

"""

二，发送 HTTP 请求，通过 Python 的 requests 库访问本地 Ollama 服务

三，检查响应并提取回答



answer\_question()：串联完整 RAG 流程

①检查问题是否为空

if not query.strip() 会过滤空字符串和只有空格的输入，避免无效查询。

② 检索相关文本

retriever.search(query, top\_k=top\_k) 调用已有的 BM25 检索器，获得排名靠前的文本块。

③ 判断有没有检索结果

如果结果为空，就直接返回提示信息和空来源列表，不必调用大模型。

④ 构造上下文并生成回答

有检索结果时，先调用 build\_context()，再调用 generate\_answer()。

⑤ 组织返回数据

返回用户问题、生成答案和 sources 来源列表，供后续 API 或网页使用。

