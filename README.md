# <b>PDF Chatbot using Python</b>
---
#### Python project that leverages [OpenAI](https://openai.com/blog/openai-api) models and [LangChain](https://python.langchain.com/docs/get_started/introduction) to query user-provided documents, creating a specialized Chatbot Assistant.
---
## 🚀 QuickStart
<sup>Note: some dependencies require Python version 3.11 or older</sup>

Ensure that Python is installed on your pc.
Clone the repository, running the following command on your terminal (make sure to have [git](https://git-scm.com/downloads) installed as well):

```bash
git clone https://github.com/REPONAME
```

Once you set up your local repository, navigate to the root folder and install the required packages, running:

```bash
pip install -r requirements.txt
```

Configure your *[OpenAI API key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key)* environment variable. Create a ".env" file in the root directory and paste in your key:
```python
OPENAI_API_KEY = "your API key goes here"
```

Finally, run one of the "chatbot" python files from the terminal.

```bash
python chatbotv2.py
```

All set! You can now interact with the chatbot. I provided a [document](https://cagayanvalley.da.gov.ph/wp-content/uploads/2018/02/Tomato.pdf) on tomato production by the [Cagayan Valley Department of Agriculture](https://cagayanvalley.da.gov.ph/), which was used to test the english version of the chatbot. 

---

(Optional): To feed your own pdf documents to the assistant, you'll need to store them in the "documents" folder and change the file name in the main python files. For this last step, go to the "chatbot.py" and "chatbotv2.py" files and set the filename to match your document's name. Note: pdfminer was used for this project, but you can adjust the text extraction process to your own needs. Try to stick with typed documents, since OCR pdfs are probably not going to work.

---

## ⚡ Motivation

Although this project is part of an ongoing investigation at *[Facultad de Ciencas Económicas Uncuyo](https://fce.uncuyo.edu.ar/)*, I decided to upload the files to github because I believe it can be helpful as a demonstration of a practical and more realistic application.

There are plenty of helpful resources online showing how to implement Langchain and OpenAI to create all sorts of chatbots. However, when starting myself with the libraries I found little walkthroughs and implementations with unique use cases . Most articles and tutorials I encountered followed the documentation very closely, so I thought it made sense to upload my own work and show a tailored implementation. Additionally, I gained a few insights while testing the bots which hopefully I'll be able to share.

More details on the investigation will be provided once it concludes.

<sup>*Special thanks to Mariano Campanello, who helped me with testing and gave me many excellent ideas.*

---

## 💡 How it works 

### Process Overview
1. **PDF Document Reading:**
   - Utilizes [PDFMiner](https://pypi.org/project/pdfminer/) to read the content of PDF documents.

2. **Content Processing:**
   - Splits the document content into "chunks" for further processing.

3. **Embeddings Creation:**
   - Converts text chunks into embeddings, representing words in vector form.

4. **Local Database Storage:**
   - Stores text pieces and their corresponding embeddings in a local database ([ChromaDB](https://docs.trychroma.com/) in version 2).

#### Additional resources
For a deeper understanding of creating conversational chatbots, consider referring to these resources:
* [Custom Knowledge Chatbot](https://www.freecodecamp.org/news/langchain-how-to-create-custom-knowledge-chatbots/)
* [Langchain Chatbots](https://python.langchain.com/docs/use_cases/chatbots/)


#### Comments on version 2 
This project in particular tries to implement some custom-made solutions when possible. In version 2, the prompt and conversation history are managed exclusively using self-written code.

See [requirements.txt](requirements.txt) for dependencies.

---

## 🌱 How to contribute

Several aspects of the chatbot could be improved, as well as the code structure itself. While building it, I've focused on the main functionalities (simple text extraction, embeddings, and managing the conversation between the user and the AI). And while the research isn't over and I'm still trying to improve the assistant's performance, there are other features and improvements I'd like to implement. These include:
* A GUI (as of now, all interactions are conducted through the terminal).
* Refining the "history" part of the window memory object.
* Implementing a way to set all configuration variables together.
* Breaking down the code into modules a bit more (to make certain functionalities accessible outside the main files).
* Improving error handling and detection of unwanted behaviour from the AI.

If you sense an opportunity to collaborate in these or other aspects of the project, or the documentation itself, feel free to fork the repository and create a pull request. Additionally, don't hesitate to reach out if you have ideas or changes that could add value. I welcome constructive criticism and suggestions. 

---

## 📝 License

This project is licensed under the MIT License


