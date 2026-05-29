from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
# ------ models
model=ChatGroq(model='llama-3.1-8b-instant')
embedding_fn=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")





vectorstore=Chroma(
    collection_name='chat_history',
    embedding_function=embedding_fn,
    persist_directory='./memory'

)
chat_history=[
    SystemMessage(content='you are sheldon cooper')
]
ragstore=Chroma(
    collection_name='rag_notes',
    embedding_function=embedding_fn,
    persist_directory='./rag_memory'
)
#print("ragstore count:", ragstore._collection.count())
#ragstore.add_documents(chunks)
#ragstore.reset_collection()
if ragstore._collection.count() == 0:
    # loading the rag notes
    loader=TextLoader(r"C:\Users\medha\OneDrive\Desktop\langchain\prompts.py\chat_history.txt")
    pages=loader.load()

    #splitting the loaded documents
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks=splitter.split_documents(pages)

    ragstore.add_documents(chunks)
    print("notes loaded into ragstore")
else:
    print("ragstore already loaded:", ragstore._collection.count(), "chunks")
    
while True:
    user_input=input('you:')
    
    if user_input=='exit':
        break
    vectorstore.add_texts(
    texts=[user_input],
    metadatas=[{'role':'user'}])

    
    relevant_memory=vectorstore.similarity_search_with_score(user_input,k=2)

    rag_results=ragstore.similarity_search_with_score(user_input,k=2)
    #print("rag scores:", [(doc.page_content[:50], score) for doc, score in rag_results])
    for r,score in relevant_memory:
        #print(r.page_content,'||',r.metadata)
        if score<0.5:



            chat_history.append(HumanMessage(content=r.page_content))
    for doc,score in rag_results:
        if score<1.5:
            chat_history.append(HumanMessage(content=doc.page_content))
    for msg in chat_history:
        print(f"{type(msg).__name__}: {msg.content}")
    '''print("\n--- context being injected ---")
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            print("HumanMessage:", msg.content[:100])
    print("------------------------------\n")'''
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI:',result.content)
    vectorstore.add_texts(texts=[result.content], metadatas=[{"role": "ai"}])
    
#print(chat_history)  



print('vectorstore ready',vectorstore._collection.count(),'messages stored')



