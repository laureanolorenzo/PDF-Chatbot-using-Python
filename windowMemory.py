class SimpleWindowMemory:
    '''Object for storing and updating both the prompt and conversation history that are passed to the OpenAI API
    to provide context to a language model (LM). It stores the prompt in string format, and it can be accessed to 
    send it to the API. It works by inserting pieces of text in between the conversation (history), the user's question,
    and the matching chunk(s) (context). The functionality of "chat completions" from the API could eventually be leveraged, providing the history and context 
    directly in the. This is likely to be developed in a future version of this project.'''
    def simple_spell_check(self,str):
        str = ' '.join(str.split())
        if not str[-1] in ['.','?','!',';',',']: 
            str = str+'.'
        return str
        

    def __init__(
            self, #default variables. Adjust to your needs
            first_time_inst = '''You are an expret AI assistant who helps users.The user will ask a question ("QUESTION"),
        which you'll have to answer based only on the text "DOCUMENT", which might hold the answer to the question.''',
            history_intro = '''You'll be shown the variable "HISTORY", which contains your previous conversation with the user.''',
            post_history_inst = 'Continue the conversation but (IMPORTANT), you can only answer based on "HISTORY" and "DOCUMENT".',
            pre_question_inst = '''Answer only getting information from "DOCUMENT", but make it seem like you knew beforehand (just casually answer the question).''',
            final_inst = '''Remember: you can only reply if the answer is explicitly in "DOCUMENT". Otherwise, just say something like "Sorry, I can't help you with that".
        It is crucial that you don't answer questions unrelated to the document! Finally, don't quote the "context" or "text" speak naturally without revealing where you got the information.'''
            ): 
        '''
        Memory window instance. This object is designed to manage the prompt and history, which are passed to the API as messages.
        Ideally it would be revised, especially since it doesn't quite adjust to the "roles" dynamic that the chat completion provides.
        '''
        self.first_time_inst = first_time_inst
        self.history_intro = history_intro
        self.post_history_inst = post_history_inst
        self.pre_question_inst = pre_question_inst #(especially this part) affects the response in terms of the "source" of information quoted
        self.final_inst = final_inst
        self.prompt = ''
        self.history = list()
        
    def update_history(self,user_question:str,answer:str):
        '''Simply appends a dictionary with roles as keys and text as values to the history'''
        user_question = f'User: {self.simple_spell_check(user_question)}'
        answer = f'AI assistant (you): {self.simple_spell_check(answer)}'
        self.history.append(dict(user_question=user_question,answer=answer))

    def clear_history(self):
        '''Not used currently'''
        self.history = list()

    def limit_window(self,n): 
        '''Method to keep the history length below the desired maximum. ''' 
        while len(self.history) > n:
            self.history.pop(0)
    def get_history(self):
        '''Turns the history "list" to string and returns it for inserting it into the prompt'''
        return ' '.join([f'{interaction["user_question"]} {interaction["answer"]}' for interaction in self.history])
    
    def update_prompt(self,context,question): 
        '''Inserts the current question and context into the prompt.'''
        question = f'"QUESTION": ###{self.simple_spell_check(question)}.###ONLY ANSWER BASED ON "DOCUMENT" ### Your answer (informative and natural):'
        context = F'"DOCUMENT": ###{self.simple_spell_check(context)}###'
        if len(self.history): # History is updated before calling this method, so if it is empty, either this is the first conversation,
            #or it's been decided that no history should be stored.
            history_text = f'"HISTORY": {self.get_history()}'
            prompt_components = [self.first_time_inst,self.history_intro,history_text,self.post_history_inst]
            self.prompt = ' '.join(prompt_components)
        else:
            self.prompt = self.first_time_inst #Sin historial
        self.question = ' '.join([self.pre_question_inst,context,self.final_inst,question])
    def update_window(self,new_question:str,context:str,previous_answer:str = None, previous_question:str = None,n:int = 1): #No sabia cómo nombrarla
        ''' 
            Main method. Manages history and prompt according to the specified values.
            ==========================================================================
            parameters:
            -new_question: String. Required. The current user's question.
            -context: String. Required. The relevant chunk from the database.
            -previous_question, previous_answer: String. Optional. If both are passed, history will be updated and added to the prompt.
            -n: Int. Optional. Default 1. Number of interactions to keep in history. If set to 0, no history will be kept. 
        '''
        if (previous_question and previous_answer): #Si hubo una interaccion previa entre el humano y la IA, que debe guardarse en el historial
            self.update_history(previous_question,previous_answer)
        self.limit_window(n)
        self.update_prompt(context,new_question)
    def get_current_prompt(self):
        # Returns prompt in string format
        return self.prompt
    def get_question(self):
        # Returns question in string format. This method (and object) was added because I needed to pass both a prompt (system)
        # and a question (user) to the chat completions endpoint. 
        return self.question





if __name__ == '__main__':  #Testing
    test_window = SimpleWindowMemory()
    while True:
        query = ''
        if query in ['Exit']:
            break
        query = input('Question')
        print('Prompt: ')
        print(test_window.get_current_prompt())
        print('History: ')
        print(test_window.get_history())
        print('-'*25, '/n')

        test_window.update_window(new_question=query,context= '<-CONTEXT->'*30)
        print('Prompt: ')
        print(test_window.get_current_prompt())
        print('History: ')
        print(test_window.get_history())

        print('-----------------------')
        print(test_window.get_question())
        
    for n, text, metadata in enumerate(zip(['A','B'],['Z','Y'])):
        print(f'{n} Primer el: {text} Segundo: {metadata}')


