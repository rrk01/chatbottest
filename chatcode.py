from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

loader = UnstructuredPDFLoader("C:/Users/ryanr/vscodeprojects/chatbottest/data/gun_swap_goddess.pdf")
data = loader.load()

print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')

OPENAI_API_KEY = '...'
PINECONE_API_KEY = '...'
PINECONE_API_ENV = '...'

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchain2"

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

query = "What is the collect stage of data maturity?"
docs = docsearch.similarity_search(query, include_metadata=True)

chain.run(input_documents=docs, question=query)