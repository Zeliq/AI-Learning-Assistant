import os
file_uploaded_state = False
from typing import TypeVar
from langchain.embeddings import HuggingFaceEmbeddings#, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import gradio as gr
from transformers import AutoTokenizer

# Alternative model sources
from ctransformers import AutoModelForCausalLM

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

# import chatfuncs.ingest as ing

##  Load preset embeddings, vectorstore, and model

embeddings_name = "BAAI/bge-base-en-v1.5"

def load_embeddings(embeddings_name = "BAAI/bge-base-en-v1.5"):

    embeddings_func = HuggingFaceEmbeddings(model_name=embeddings_name)

    global embeddings

    embeddings = embeddings_func

    return embeddings

def get_faiss_store(faiss_vstore_folder,embeddings):
    import zipfile
    with zipfile.ZipFile(faiss_vstore_folder + '/' + faiss_vstore_folder + '.zip', 'r') as zip_ref:
        zip_ref.extractall(faiss_vstore_folder)

    faiss_vstore = FAISS.load_local(folder_path=faiss_vstore_folder, embeddings=embeddings)
    os.remove(faiss_vstore_folder + "/index.faiss")
    os.remove(faiss_vstore_folder + "/index.pkl")
    
    global vectorstore

    vectorstore = faiss_vstore

    return vectorstore

import chatfuncs.chatfuncs as chatf

chatf.embeddings = load_embeddings(embeddings_name)
chatf.vectorstore = get_faiss_store(faiss_vstore_folder="faiss_embedding",embeddings=globals()["embeddings"])

def load_model(model_type, gpu_layers, gpu_config=None, cpu_config=None, torch_device=None):
    print("Loading model")

    # Default values inside the function
    if cpu_config is None:
        cpu_config = chatf.cpu_config


    hf_checkpoint = 'declare-lab/flan-alpaca-large'#'declare-lab/flan-alpaca-base' # # #
    
    def create_hf_model(model_name):

        from transformers import AutoModelForSeq2SeqLM,  AutoModelForCausalLM
        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = chatf.context_length)

        return model, tokenizer, model_type

    model, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

    chatf.model = model
    chatf.tokenizer = tokenizer
    chatf.model_type = model_type

    load_confirmation = "Finished loading model: " + model_type

    print(load_confirmation)
    return model_type, load_confirmation, model_type

model_type = "Flan Alpaca (small, fast)"
load_model(model_type, 0, chatf.cpu_config)

def docs_to_faiss_save(docs_out:PandasDataFrame, embeddings=embeddings):

    print(f"> Total split documents: {len(docs_out)}")

    print(docs_out)

    vectorstore_func = FAISS.from_documents(documents=docs_out, embedding=embeddings)
    chatf.vectorstore = vectorstore_func

    out_message = "Document processing complete"

    return out_message, vectorstore_func

 # Gradio chat

block = gr.Blocks(theme = gr.themes.Soft())

with block:
    ingest_text = gr.State()
    ingest_metadata = gr.State()
    ingest_docs = gr.State()

    model_type_state = gr.State(model_type)
    embeddings_state = gr.State(globals()["embeddings"])
    vectorstore_state = gr.State(globals()["vectorstore"])  

    model_state = gr.State() # chatf.model (gives error)
    tokenizer_state = gr.State() # chatf.tokenizer (gives error)

    chat_history_state = gr.State()
    instruction_prompt_out = gr.State()

    gr.Markdown("<h1><center>TextBook AI</center></h1>")        
    
    gr.Markdown("Chat with textbook documents. This is a small model (Flan Alpaca), that can only answer specific questions that are answered in the text. It cannot give overall impressions of, or summarise the document.\n\n Please note that LLM chatbot may give incomplete or incorrect information, so please use with care.")

    with gr.Row():
        current_source = gr.Textbox(label="Current data source(s)", value="Lambeth_2030-Our_Future_Our_Lambeth.pdf", scale = 10,visible=False)
        current_model = gr.Textbox(label="Current model", value=model_type, scale = 3,visible=False)
    
    with gr.Tab("Chatbot"):

        with gr.Row():
            #chat_height = 500
            chatbot = gr.Chatbot(avatar_images=('user.gif', 'bot_icon.gif'),bubble_full_width = False, scale = 1) # , height=chat_height
            with gr.Accordion("Open this tab to see the source paragraphs used to generate the answer", open = False):
                sources = gr.HTML(value = "Source paragraphs with the most relevant text will appear here", scale = 1) # , height=chat_height

        with gr.Row():
            message = gr.Textbox(
                label="Enter your question here",
                lines=1,
            )     
        with gr.Row():
            submit = gr.Button(value="Send message", variant="secondary", scale = 1)
  
        current_topic = gr.Textbox(label="Feature currently disabled - Keywords related to current conversation topic.", placeholder="Keywords related to the conversation topic will appear here",visible=False)
            
    with gr.Row():
        with gr.Tab("Load in a file to chat with"):
            with gr.Accordion("PDF file", open = False):
                in_pdf = gr.File(label="Upload pdf", file_count="multiple", file_types=['.pdf'])
                load_pdf = gr.Button(value="Load in file", variant="secondary", scale=0)

            ingest_embed_out = gr.Textbox(label="File preparation progress")

    # with gr.Tab("Advanced features"):
            out_passages = gr.Slider(minimum=1, value = 2, maximum=10, step=1, label="Choose number of passages to retrieve from the document. Numbers greater than 2 may lead to increased hallucinations or input text being truncated.",visible=False)
            temp_slide = gr.Slider(minimum=0.1, value = 0.1, maximum=1, step=0.1, label="Choose temperature setting for response generation.", visible=False)
        

    # Load in a pdf
    load_pdf_click = load_pdf.click(ing.parse_file, inputs=[in_pdf], outputs=[ingest_text, current_source])\
        .then(ing.text_to_docs, inputs=[ingest_text], outputs=[ingest_docs])\
        .then(docs_to_faiss_save, inputs=[ingest_docs], outputs=[ingest_embed_out, vectorstore_state])\
        .then(chatf.hide_block)
    file_uploaded_state= True

    if file_uploaded_state:
    # Click/enter to send message action
        response_click = submit.click(chatf.create_full_prompt, inputs=[message, chat_history_state, current_topic, vectorstore_state, embeddings_state, model_type_state, out_passages], outputs=[chat_history_state, sources, instruction_prompt_out], queue=False, api_name="retrieval").\
                    then(chatf.turn_off_interactivity, inputs=[message, chatbot], outputs=[message, chatbot], queue=False).\
                    then(chatf.produce_streaming_answer_chatbot, inputs=[chatbot, instruction_prompt_out, model_type_state], outputs=chatbot)
        response_click.then(chatf.add_inputs_answer_to_history,[message, chatbot, current_topic], [chat_history_state, current_topic]).\
                    then(lambda: chatf.restore_interactivity(), None, [message], queue=False)
block.queue(concurrency_count=1).launch(debug=True)
# -