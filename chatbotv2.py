#Imports 
from dotenv import load_dotenv
from windowMemory import SimpleWindowMemory #Custom self-made memoty window 
from pdfminer.high_level import extract_text #Again, can improve a lot in terms of pdf processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
import os
from openai import OpenAI
import time
import chromadb
import numpy as np
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import tiktoken






load_dotenv() #Loading the API key. Remember to create a '.env' file containing the following: "OPENAI_API_KEY = <your key>"



################################4

###Config Variables###
# IMPORTANT: their optimal values depend on the type of text that is being processed. 
# For instance: for news articles, smaller chunk sizes and passing more individual chunks to the API works better (for me at least).
# Text quality is also key. From the research, I've concluded that writing your own texts with simple and explicit information works best.

################################


file_name = 'tomato_production' #Remember to change this variabke if you supply your own text
# Will serve as an identifier in the ChromaDB, (if the collection doesn't already contain embeddings with this filename 
# in their metadata, then the text extraction and embeddings creation process is started).
file_path = f'documents//{file_name}.pdf' 


chunk_size = 100
chunk_overlap = 10 
window_len = 0 # "n". Controls the length of the conversation. If set to 0, a new conversation is created each time
# After some testing, I found a trade-off between being able to keep a history of the conversation, and 
# getting responses that are consistent with the context



n_results_passed = 4 #How many chunks are to be passed to the API. Be mindful when adjusting, since it affects token consumption
                    # Multiple smaller chunks worked best with the "tomato" pdf
n_results_extra = 2 #Not too important. Controls the number of "extra" chunks (lower scoress) that are shown after a response


### Threshold (important)
metric_threshold = 0.82 # If in any given interaction no chunk  reaches this degree of similarity to the question, then
# the API call is canceled and an automatic response is sent instead (serves as a filter). If you wish to always
# call the API, then set it to 0

###Another layer of protection against the chatbot responding unrelated questions
second_filter = True #I still had a hard time limiting responses to the context. So this should help ensure adequate behaviour
def classification_filter(question,answer,context):
    return {'system':f''''You determine if an answer to a question is based ONLY on the provided context. Not just related to the context, but built on it.''',   
    'user': '''The question is: ##{question}## The answer is ##{answer}## and the context is ##{context}##.
    Is the answer entirely based on the context, or does it come from elsewhere? say yes if they are COMPLETELY realted, no otherwise. If the question is ambiguous or tricky or unrelated, also say no. Your answer'''}

path_to_save_to = 'chromaindex/' #The folder where the chromadb containing the embeddings is stored

embeddedings_model='text-embedding-ada-002' #Standard embeddings model
completions_model = 'gpt-3.5-turbo' #Completions model

embeddings_encoding = tiktoken.encoding_for_model(embeddedings_model)
completion_encoding = tiktoken.encoding_for_model(completions_model)

#####Token consumption#####




### Functions ###
### This section is meant to help keep the main loop short and simple, but I need to redesign the whole code to achieve that.
### Although it is an improvement from v1, I need to tidy up the code and clearly define classes (maybe a "chain" object). 
### Hopefully I can do so in future versions.
def num_tokens_from_string(
        string: str,
        encoding = tiktoken.encoding_for_model('text-embedding-ada-002'),
        encoding_name: str = None
        ) -> int: # Realized the "chunk_size" parameter of text splitter works with characters 
                  # by default, so a function to convert words to tokens is needed if we want the chunk size to be measured in tokens
    if (encoding_name):
        encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



token_mapping = { ###Per token prices. Used for calculating costs
    embeddedings_model: 0.00010/1000, 
    f'{completions_model}_input': 0.0005/1000,
    f'{completions_model}_output': 0.0015/1000,
    f'{completions_model}_prompt_check': 0.0005/1000,
    f'{completions_model}_check_response': 0.0005/1000,
}
def calculate_total_cost(token_consumption): 
    return "{:.6f}".format(np.sum([token_mapping[k]*token_consumption[k] for k in token_consumption.keys()]))

#Show context if answer isn't in document
def show_context(query_results,n_results_extra,choice):
    
    if choice in ['yes','yess','Yes','YES', 'y','Y']:
        print('*-----------------------')
        print('Relevant documents:')
        for n, (text, metadata,distance) in enumerate(zip(query_results['documents'][0][:n_results_passed],query_results['metadatas'][0][:n_results_passed],query_results['distances'][0][:n_results_passed])): 
            text = ''.join([s for s in text.splitlines(True) if s.strip('\r\n')])
            print(f'{n+1}:\n{text}\nScore: {1-distance}\nAdditional data: {metadata}')
        if n_results_extra > 0: #Useful to see resulting chunks when searching for optimal parameter configuration
            print('Might be of interest: ')
            for n, (text, metadata) in enumerate(zip(query_results['documents'][0][n_results_passed:n_results_passed+n_results_extra],query_results['metadatas'][0][n_results_passed:n_results_passed+n_results_extra])): 
                text = ''.join([s for s in text.splitlines(True) if s.strip('\r\n')])
                print(f'{n+1}:\n{text}\nAdditional data: {metadata}')

def get_usage(token_consumption):
        usage = ''
        usage += '*-----------------------'
        usage += '\nToken usage: '
        usage += f'\nEmbeddings tokens: {token_consumption[embeddedings_model]}'
        if f"{completions_model}_input" in token_consumption:
            usage += f'\nPrompt tokens: {token_consumption[f"{completions_model}_input"]}'
            usage += f'\nCompletion tokens: {token_consumption[f"{completions_model}_output"]}'
        if f"{completions_model}_prompt_check" in token_consumption:
            total_check_tokens = token_consumption[f"{completions_model}_prompt_check"] + token_consumption[f"{completions_model}_check_response"]
            usage += f'\nCheck tokens: {total_check_tokens}'
        usage += f'\nTotal costs: US${calculate_total_cost(token_consumption)}\n'
        usage += '-----------------------------------------------\n'*2
        return usage



#This is just so that chromadb knows which token counting function to use
embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=embeddedings_model)





###Text splitting
character_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
    length_function = num_tokens_from_string,
    separators= ['\n ', '\n\n','\n',' ','']

)

### Not too important. This variables tries to limit the number of times the main loop runs with errors.
max_n_tries = 3 

## chromaDB client and collection setup


client = chromadb.PersistentClient(path=path_to_save_to)
collection_name = 'FCE_chatbot_collection' 
collection = client.get_or_create_collection(collection_name,
                                embedding_function=embedding_function,
                                metadata={"hnsw:space": "cosine"}) #Experiment with this, but keep scoring in mind.


embeddings = OpenAIEmbeddings(model=embeddedings_model) 

##Checking if the document is already in the collection
doc_exists = len(collection.get(where={'doc_name':file_name})['ids']) 
if not doc_exists: 
    try:
        text = extract_text(file_path) 
        # It handled the tomato file pretty well, but it couldn't deal with table in the earnings report.
    except:
        print('Error occurred while processing pdf. Please try again using a different file. ')
        quit() #Not sure if this is the right approach
    text = text.replace('\n\n','\n') #Some pdfs have multple line breaks together. This helps a bit 
    chunks = character_splitter.split_text(text)
    embedded_docs = embeddings.embed_documents(chunks,chunk_size=chunk_size) 
    doc_ids = [f'{str(time.time())}-{str(x)}' for x in np.arange(0,len(chunks))]#Just a simple solution to create unique identifiers for each chunk

    collection.add(
    documents=chunks,
    embeddings=embedded_docs,
    ids=doc_ids,
    metadatas=[{'doc_name':file_name,'Chunk_length':len(x),'Chunk_size':num_tokens_from_string(x,embeddings_encoding)} for x in chunks] #Ideally you would have metadata about page number, paragraph, etc.

    )
    print(f'Embeddings created successfully for "{file_name}".')
else:
    print(f'Embeddings for "{file_name}" already exist.')


if __name__ == '__main__': 
    openai_client = OpenAI()
    window_memory = SimpleWindowMemory()
    token_consumption = dict()
    query = ''
    context = ''
    number_of_tries = 0
    print('\n'+'-'*25+ '\n')
    previous_answer = None #To start the loop without a history
    previous_question = None 
    n_substitution_chunks = 3 # Returns the "n" most relevant chunks when there are no matches
    while True: #Main Loop
        context = ''
        print('Type "Exit" or "Quit" to leave this chat.' )
        query = input('>>>Your question: ')
        if query in ['exit','EXIT','Exit', 'Quit', 'quit', 'q', 'QUIT', 'Q']: 
            break
        try:
            query_embeddings = embeddings.embed_query(query)
            n_results = n_results_passed + n_results_extra
            if n_results > 10:
                n_results = 10 #10 is enough
            query_results = collection.query(query_embeddings=query_embeddings,n_results=n_results) # The data in the response is a bit tricky to access!
            if token_mapping[embeddedings_model]:
                token_consumption[embeddedings_model] = num_tokens_from_string(query,embeddings_encoding) #Costs
        except:
            number_of_tries +=1
            if number_of_tries < max_n_tries:
                print('*-----------------------')
                print('Please try again.')
                continue
            else:
                print('*-----------------------')
                print('Attempt limit reached. Please restart the program.')
                break
        
        print('*-----------------------')
        if not query_results['ids']: #This is for when we get an empty result.
            print('*-----------------------')
            print('It seems there are no answers for your question in the document. Please try again with a different question.')
            if token_consumption[embeddedings_model]:
                    print(get_usage(token_consumption))
            time.sleep(5)
            continue
        
        result_metric = 1 - query_results['distances'][0][0] 
        if result_metric < metric_threshold and not previous_answer: #Extra filter to avoid unrelated answers. This is not implemented in version 1
            print('It seems there are no answers for your question in the document. Please try again with a different question.') #Could change the message here
            time.sleep(5)
            if token_consumption[embeddedings_model]:
                print(get_usage(token_consumption))
            print('*-----------------------')


            print('')
            
            if n_substitution_chunks > 0: #"Replacement" chunks
                time.sleep(3) 
                print('Might be of interest: ')
                for n, (text, metadata,distance) in enumerate(zip(query_results['documents'][0][:n_substitution_chunks],query_results['metadatas'][0][:n_substitution_chunks],query_results['distances'][0][:n_substitution_chunks])): #chunks como "reemplazo" de la repuesta
                    text = ''.join([s for s in text.splitlines(True) if s.strip('\r\n')])
                    print(f'{n+1}:\n{text}\nScore: {1 - distance}\nAdditional data: {metadata}')
                    print(f'Score: {calculate_total_cost(token_consumption)}')
            print('FIRST FILTER ACTIVATED')
            continue
        ## If this stage is reached, it means the context is relevant to the question, so the API is called
        for n,text in enumerate(query_results['documents'][0][:n_results_passed]): #Add only first "n" chunks to context 
            text = ''.join([s for s in text.splitlines(True) if s.strip('\r\n')]) # To further remove white space. Maybe not neededs
            context += '#Piece of context:' + text + '\n' 
        window_memory.update_window(
            new_question=query,
            context=context,
            previous_answer=previous_answer,
            previous_question=previous_question,
            n = window_len
            )
        completion = openai_client.chat.completions.create(
            model=completions_model,
            temperature=0.05,
            n = 1, #Number of responses from the API (read the docs for more info). One is more than enough here
            max_tokens=500, 
            messages= [ #Key variable. In this project, the whole prompt (including history) is passed in "system" and "user".
                #But it should be passed as the conversation progresses 
                {'role':'system', 'content': window_memory.get_current_prompt()},
                {'role':'user', 'content': window_memory.get_question()}
            ]
        )  
        completion = completion.dict() #The completion object needs to be converted to dict in order to access "usage"
        try:
            answer = completion['choices'][0]['message']['content'] #What is then given to the user as response 
            token_consumption[f'{completions_model}_input'] = completion['usage']['prompt_tokens'] #Keeping track of costs
            token_consumption[f'{completions_model}_output'] = completion['usage']['completion_tokens']
        except:
            print('Connection with OpenAI API failed. Please try again')
            continue
        ####
        ## Second filter to check if response makes sense (is actually related to context)
        ###
        if second_filter:
            completion_check = openai_client.chat.completions.create(
            model=completions_model,
            temperature=0.05,
            n = 1, #Number of responses from the API (read the docs for more info). One is more than enough here
            max_tokens=500, 
            messages= [ #Key variable. In this project, the whole prompt (including history) is passed in "system" and "user".
                #But it should be passed as the conversation progresses 
                {'role':'system', 'content': classification_filter(query,answer,context)['system']},
                {'role':'user', 'content': classification_filter(query,answer,context)['user']}
            ] 
            )
            completion_check = completion_check.dict()
            completion_check_msg = completion_check['choices'][0]['message']['content']
            token_consumption[f'{completions_model}_prompt_check'] = completion_check['usage']['prompt_tokens']
            token_consumption[f'{completions_model}_check_response'] = completion_check['usage']['completion_tokens']  
            
            if completion_check_msg.lower() == 'no':
                print('*-----------------------')
                print('It seems there are no answers for your question in the document. Please try again with a different question.')
                time.sleep(3)
                completion_check = completion_check
                print('CHECK',completion_check_msg)
                time.sleep(3)
                choice = input('Show relevant pieces of text? (yes/no)')
                show_context(query_results,n_results_extra,choice)
                continue
                
        print('*-----------------------')
        print(f">>>AI's response : {answer}")
        time.sleep(5) # Gives the user some time to read the answer
        if (window_len > 0):
            previous_answer = answer #This sets the variable for the next iteration, where it will be added to the history
            previous_question = query
        print(get_usage(token_consumption))
        choice = input('Show relevant pieces of text? (yes/no): ')
        show_context(query_results,n_results_extra,choice)
    print('Conversation ended successfully. See you soon!')

