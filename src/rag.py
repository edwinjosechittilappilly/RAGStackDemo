import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from pydantic.v1 import BaseModel, Field
from typing import List
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model_name = "gemini-pro"

llm = GoogleGenerativeAI(model=model_name,
                         google_api_key=GOOGLE_API_KEY,
                         max_output_tokens=2048,
                         max_retries=20,
                         temperature=0)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")



ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE_Philo = os.getenv("ASTRA_DB_KEYSPACE_Philo")


vstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="astra_vector_demo",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_KEYSPACE_Philo,
)
retriever_vector_score = vstore.as_retriever()

metadata_field_info = [
    AttributeInfo(
        name="author",
        description="The Author of the Quote, if no author name then keep it empty",
        type="string",
    ),
    AttributeInfo(
        name="tags",
        description="Further information about the quote, like a tag or category, if no tag then keep it empty",
        type="string or list[string]",
    ),
]
document_content_description = "The text of the quote from the philosopher."
retriever_self_query = SelfQueryRetriever.from_llm(
    llm, vstore, document_content_description, metadata_field_info, verbose=True
)



class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str):
        """
        Parses the given text and returns a LineList object.

        Args:
            text (str): The text to be parsed.

        Returns:
            LineList: The parsed LineList object.
        """
        # print(f'parsing text: {text} \n')
        lines = text.strip().split("\n")
        lines = [
            line for line in lines if line.strip()
            and "alternative" not in line and "original question" not in line
        ]
        print(f'lines: {lines}')
        return lines


output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate two 
    different versions of the given user question to retrieve relevant quotes from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. Do not change the meaning of the question, do not change the authors name or the tag
    Provide these alternative questions separated by newlines. also mention the original question. And have question vartion by adding the author name as author: name of author in lower case.
    Original question: {question}""",
)

# Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)


retriever_multi_query = MultiQueryRetriever(
    retriever=retriever_self_query, llm_chain=llm_chain, parser_key="lines", include_original=True)

template = """Human: Greet the user politely, is its a generic conversation. If you don't know the answer, just say that you don't know, don't try to make up an answer. Please make sure the answer is short and precise.
                Always Answer with respect to the context or with respect to the documents  given in the context. Note that you are a Copilot  for a Philospoher, that helps the user in answer queries about quotes of philosophers. 
                Also if not enought context please do not mention Unfortunately I do not have enough context, instead answer with the context available. 
                Make the answer in a convertional way. if context is empty, then tell the user that you dont know the asnwer and give an alternative question to the user.
                Context: {context}
                Question: {question}
                Assistant:"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    print(f"docs: {docs}")
    try:
        return "\n\n".join(doc.page_content for doc in docs)
    except:
        return "NoData"



rag_chain_from_docs = (RunnablePassthrough.assign(
    context=(lambda x: format_docs(x["context"])))
                        | prompt
                        | llm
                        | StrOutputParser())

rag_chain_with_source_multi= RunnableParallel({
    "context": retriever_multi_query,
    "question": RunnablePassthrough()
}).assign(answer=rag_chain_from_docs)



rag_chain_with_source_basic = RunnableParallel({
    "context":
    retriever_vector_score,
    "question":
    RunnablePassthrough()
}).assign(answer=rag_chain_from_docs)


# Other inputs
# question =
if __name__ == "__main__":
    question = "give a quote by plato"
    print(f"Question: {question}")
    try:
        response = rag_chain_with_source_multi.invoke(question)
    except Exception as e:
        print(f"Error: {e}")
        response = rag_chain_with_source_basic.invoke(question)

    print(response)
