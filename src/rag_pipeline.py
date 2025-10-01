# Install the libraries needed for the project
import subprocess
import pkg_resources

def install_if_needed(package, version):
    '''Function to ensure that the libraries used are consistent to avoid errors.'''
    try:
        pkg = pkg_resources.get_distribution(package)
        if pkg.version != version:
            raise pkg_resources.VersionConflict(pkg, version)
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
        subprocess.check_call(["pip", "install", f"{package}=={version}"])

install_if_needed("langchain-core", "0.3.72")
install_if_needed("langchain-openai", "0.3.28")
install_if_needed("langchain-community", "0.3.27")
install_if_needed("unstructured", "0.18.11")
install_if_needed("langchain-chroma", "0.2.5")
install_if_needed("langchain-text-splitters", "0.3.9")

# Import the required packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os

# Load the HTML as a LangChain document loader
loader = UnstructuredHTMLLoader(file_path="data/mg-zs-warning-messages.html")
car_docs = loader.load()

# Load the models required to complete the exercise
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# Initialize RecursiveCharacterTextSplitter to make chunks of HTML text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the car documents with text_splitter
splits = text_splitter.split_documents(car_docs)

# Initialize Chroma vectorstore with documents as splits and using OpenAIEmbeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Setting up vectorstore as retriever
retriever = vectorstore.as_retriever()

# Define RAG prompt
prompt = ChatPromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:")

# Setup the chain
rag_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Initialize query
query = "The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"

# Invoke the query
answer = rag_chain.invoke(query).content
print(answer)