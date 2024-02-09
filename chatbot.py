
#Imports
from dotenv import load_dotenv
from langchain.docstore.document import Document
from pdfminer.high_level import extract_text #Very simple PDF processing. Can be expanded a lot.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
import os
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate
import time
import tiktoken


load_dotenv() #Loading the API key. Remember to create a '.env' file containing the following: "OPENAI_API_KEY = <your key>"

### Functions ###
def num_tokens_from_string(
        string: str,
        encoding = tiktoken.encoding_for_model('text-embedding-ada-002'),
        encoding_name: str = None
        ) -> int: #Not used too much in this version. Realized the "chunk_size" parameter of text splitter works with characters 
                  # by default, so a function to convert words to tokens is needed if we want the chunk size to be measured in tokens
    if (encoding_name):
        encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



###Text extraction from PDF

file_path = 'documents//tomato_production.pdf' #Or a custom name if another document is used
file_path = file_path.replace(r'//',"\\")
file_name = file_path.split('\\')[-1]
if file_name[-4:] == '.pdf':
    file_name = file_name[:-4] #To check if the embeddings already exist
text = extract_text(file_path)

### Splitting the text in chunks. This step is crucial, since it highly affects the quality of the context that
# is later fed to the API. Try different values for each parameter and research further if possible

character_splitter = RecursiveCharacterTextSplitter(#These parameters can be played with. Highly dependent on the text!
    chunk_size = 500,
    length_function = num_tokens_from_string,
    chunk_overlap = 40,
    separators=['\n\n'] #Tries to capture paragraphs. Again, depends on the specifics of the pdf doc
)
chunks = character_splitter.split_text(text)


embeddings = OpenAIEmbeddings(model='text-embedding-ada-002') #Standard embeddings model (at least as of the creation of this project)
texts = character_splitter.create_documents(chunks) #Text extraction

###Checking if a file with the embeddings already exists. If it doesn't, a new one is created.
index_path = 'faiss_index/index.pkl'
if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
    print(f'Embeddings for "{file_name}" already exist.')
    vectorindex_openai = FAISS.load_local('faiss_index', embeddings)
else:
    #A way to check the embeddings costs of calling the API has not been implemented (in this first version). See "chatbotv2.py" for an implementation
    vectorindex_openai = FAISS.from_documents(texts, embeddings)
    vectorindex_openai.save_local('faiss_index')    
    print(f'Embeddings for "{file_name}" created successfully')

### Custom prompt (trying to avoid getting hallucinations)
system_template = '''
You are an AI assistant who helps users. Try to continue the following conversation:
---
{chat_history}
---
Based only on the following context, answer the user's question
---
{context}
---
The human user can also ask about your previous conversation, so keep that in mind.
If the answer to the question isn't in the context or you can't answer, just say something like "sorry, I can't help you with that". Do not formulate an answer outside of the context!
Finally, don't quote the "context" or "text" speak naturally without revealing where you got the information.
'''
user_template = "User question: {question}. AI assistant's response (natural but informative): "
messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template)
]
qa_prompt = ChatPromptTemplate.from_messages(messages)


### Window memory and relevant chunks of information
llm = ChatOpenAI(model_name= 'gpt-3.5-turbo',temperature=0.05) #El modelo que va a procesar los chunks y generar una respuesta
window_memory = ConversationBufferWindowMemory(memory_key='chat_history',
                                            llm= llm, 
                                            k = 4,input_key='question', output_key='answer',
                                            return_messages= True) 

chain = ConversationalRetrievalChain.from_llm(llm = llm,
                                            retriever=vectorindex_openai.as_retriever(search_kwargs={"k": 2,"score_threshold": .82}), #Nro de matches relevantes
                                            memory=window_memory,
                                            combine_docs_chain_kwargs = {'prompt': qa_prompt},
                                            return_source_documents = True,

                                            )

if __name__ == '__main__':
    continuar = ''
    query = ''
    print('\n'+'-'*25+ '\n')
    while True: #Main loop. Basic but serves its purpose
        print('Type "Exit" to leave this chat.' )
        query = input('>>>Your question: ')
        if query in ['exit','EXIT','Exit']: # Closes the loop
            break
        with get_openai_callback() as cb: # First solution to show token consumption. In v2 I map the prices per token to token consumption (can handle embeddings too)
            result = chain.invoke({'question':query})
            print(f">>>AI's response : {result['answer']}")
            window_memory.save_context( #Updates window memory
                {'question':query + '. ---Contexto: ' + result['source_documents'][0].page_content+ '---'},
                {'answer': result['answer']}
                )
            time.sleep(4)
            print('\n'+'-'*25 + '\n\n\n')
            print('>>Relevant pieces of information: \n')
            for n,i in enumerate(result['source_documents']): #Relevant chunks > Expanded a bit in v2
                print(f'{n+1}: \n>>>{i.page_content}"\n>>Additional data: \n{i.metadata}') #Metadata (if it exists)
            print('\n'+'-'*25+ '\n')
            print('Usage: ')
            print(cb)
        time.sleep(3)
        print('\n'+'-'*25)

