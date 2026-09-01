from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = PROJECT_ROOT / 'data' / 'company_handbook.md'
VECTOR_DIR = PROJECT_ROOT / 'db' / 'chroma.db'

DASHSCOPE_BASE_URL=os.getenv('DASHSCOPE_BASE_URL')
DASHSCOPE_API_KEY=os.getenv('DASHSCOPE_API_KEY')
DASHSCOPE_MODEL_NAME=os.getenv('DASHSCOPE_MODEL_NAME')

EMBEDDING_MODEL=os.getenv('EMBEDDING_MODEL_PATH')
RERANKER_MODEL=os.getenv('RERANKER_MODEL_PATH')

print('正在加载BGE嵌入模型')

embeddings = HuggingFaceEmbeddings(
    model_name = os.getenv('EMBEDDING_MODEL_PATH'),
    model_kwargs={'device': 'cpu'},  # 如果是 英伟达显卡可以填写 cuda，苹果M芯片填写 mps，其他填写 cpu 或这个参数都不写
    encode_kwargs={'normalize_embeddings': True}
)

print('正在加载BGE Reranker')
reranker = CrossEncoder(RERANKER_MODEL, max_length=512,)

llm = init_chat_model(
    model = DASHSCOPE_MODEL_NAME,
    api_key = DASHSCOPE_API_KEY,
    base_url = DASHSCOPE_BASE_URL,
    model_provider = 'openai'
)

#构建多路召回retriver
def build_ensemble_retriver():
    """构建bm25+vector的混合检索器"""
    if not DOC_PATH.exists():
        raise FileNotFoundError(f'找不到文件{DOC_PATH}')
    with DOC_PATH.open('r', encoding='utf-8') as f:
        markdown_text = f.read()

    headers_to_split_on = [
        ('##', 'Chapter'),
        ('###', 'Section')
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)  #确立切分规则
    md_header_splits = markdown_splitter.split_text(markdown_text)      #切分

    #第二层按照['\n\n','\n']顺序递归切分
    #第三层：为了防止某个章节依然过长，再叠加一个字符集滑动窗口
    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=['\n\n','\n']
    )
    splits = text_splitter.split_documents(md_header_splits)

    print(f'文档切分完成，共生成{len(splits)}个语义文本块。')

    #路线A：全文关键字检索（BM25）
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k=5

    #路线B向量语义检索
    if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()):
        print('检测到本地持久化向量库，直接加载')
        vectorstore =  Chroma(persist_directory=str(VECTOR_DIR),
                      embedding_function=embeddings)

    else:
        print('本地无缓存，生成向量库')
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(VECTOR_DIR),
        )

    voctor_retriever = vectorstore.as_retriever(search_kwargs={'k':5})

    #混合使用
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, voctor_retriever],
        weights=[0.4, 0.6]
    )

    return ensemble_retriever

retriever = build_ensemble_retriver()

class QueryExpansion(BaseModel):
    expanded_queries: list[str] = Field(description='从三个不同维度扩写3个相关检索词或短语')
    hypothetical_document:str = Field(description='针对该问题的一段假设性，看似专业的官方制度回答片段（允许伪造数字）')

expansion_parser = JsonOutputParser(pydantic_object=QueryExpansion)

def expand_and_hyde(original_query:str) -> list[str]:
    """利用LLM生成多维度扩写与HYDE假设"""
    prompt = ChatPromptTemplate.from_template(
        "你是一名专业的企业 HR 专家。为了提高提高知识库检索命中率，请协助处理用户的原始提问。\n"
        "任务 1（多维扩写）：站在不同视角（如政策名次、审批流程、系统操作）扩写 3 个相关检索词或短语。\n"
        "任务 2（HyDE假设）：用官方、严谨的 HR 规章制度口吻，伪造一段回答该问题的文本。不管事实是否正确，重点是极度模仿「员工手册」的很专业行文风格和词汇分布。\n\n"
        "用户原始问题：{query}\n\n"
        "{format_instructions}"
    )

    chain = prompt | llm | expansion_parser

    try:
        result = chain.invoke({
            'query':original_query,
            'format_instructions': expansion_parser.get_format_instructions()
        })

        print(f'\n原始问题: "{original_query}"')
        print(f'      ->衍生查询: {result['expanded_queries']}')
        print(f'      ->HYDE伪文: {result["hypothetical_document"]}')

        return [original_query] + result['expanded_queries'] + [result['hypothetical_document']]

    except Exception as e:
        print(f' LLM调用失败， 降级使用基础检索。原因{e}')
        return [original_query]


@tool
def search_hr_policy(query:str) -> str:
    """
    高级知识搜素引擎（具备自动改写，混合检索，重排功能）
    当用户查询任何关于公司规章制度，差旅报销标准，假期政策，福利等相关信息，必须调用此工具
    输入参数 query 必须时用户原始问题
    :param query:
    :return:
    """
    #获取五个查询变体组成的查询矩阵
    search_queries = expand_and_hyde(query)

    #多路并发检索
    all_condition_docs = []
    for q in search_queries:
        docs = retriever.invoke(q)
        all_condition_docs.extend(docs)

    #文档去重（以文档内容作为唯一标识）
    # 文档去重（以文档内容作为唯一标识）
    unique_docs = list({doc.page_content: doc for doc in all_condition_docs}.values())

    if not unique_docs:
        return '知识库未检索到相关政策，请提示用户询问HR人工'

    #cross_encoder 精准重排
    sentence_pairs = [[query,doc.page_content]for doc in unique_docs]
    scores = reranker.predict(sentence_pairs)

    scored_doc = list(zip(unique_docs, scores))
    #按模型打分从高到低排序
    scored_doc.sort(key=lambda x: x[1], reverse=True)

    #截取真正的TOP-3并返回组装文本
    top_3_docs = [doc for doc, _ in scored_doc[:3]]

    context_parts = []
    for i, doc in enumerate(top_3_docs, 1):
        chapter = doc.metadata.get('Chapter', '未知章节')
        section = doc.metadata.get('Section', '未知段落')
        context_parts.append(f'「来源 {i}」 {chapter} > {section} \n {doc.page_content}')

    merged_context = '\n\n'.join(context_parts)

    return f'「知识库检索结果」\n{merged_context}'