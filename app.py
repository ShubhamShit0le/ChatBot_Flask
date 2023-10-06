
import os
from flask import Flask, jsonify, render_template, request

# Import necessary modules for your chatbot functionality
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
from langchain import OpenAI
from llama_index import ServiceContext
import webbrowser
import skyciv
from openai import OpenAIError  

app = Flask(__name__)

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = 'Your OPENAI Api Key'

SKYCIV_API_USERNAME = "Your SKYCIV API USERNAME"
SKYCIV_API_KEY = "YOUR SKYCIV API KEY"

z_coord = None
y_coord = None
waiting_for_coordinates = False

# Define your construct_index function here
def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    chunk_overlap_ratio = 0.2
    # set chunk size limit
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    # Load the index from disk (assuming 'index.json' exists)
    index.storage_context.persist(persist_dir="./content/")

    return index

# Sample function to handle user input (you can implement SkyCiv functionality here)
def ask_bot(user_query):
    storage_context = StorageContext.from_defaults(persist_dir='./content/')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    
    try:
        response = query_engine.query(user_query)
        return response.response
    except OpenAIError as e:
        # Handle OpenAI API errors here
        return "Bot: An error occurred during the request."

# Sample function for creating SkyCiv model
def create_skyciv_model(z_coord, y_coord):
    try:
        # Initialize the SkyCiv model and perform operations here
        model = skyciv.Model("metric")
        model.nodes.add(0, 0, 0)
        model.nodes.add(float(y_coord), 0, 0)
        model.nodes.add(float(y_coord), float(y_coord), 0)
        model.nodes.add(0, float(y_coord), 0)
        model.nodes.add(0, 0, -float(z_coord))
        model.nodes.add(float(y_coord), 0, -float(z_coord))
        model.nodes.add(float(y_coord), float(y_coord), -float(z_coord))
        model.nodes.add(0, float(y_coord), -float(z_coord))

        model.members.add(1, 2, "Continuous", 1, 0, "FFFFFF")
        model.members.add(2, 3, "Continuous", 1, 0, "FFFFFF")
        model.members.add(3, 4, "Continuous", 1, 0, "FFFFFF")
        model.members.add(4, 1, "Continuous", 1, 0, "FFFFFF")
        model.members.add(1, 5, "Continuous", 1, 0, "FFFFFF")
        model.members.add(2, 6, "Continuous", 1, 0, "FFFFFF")
        model.members.add(3, 7, "Continuous", 1, 0, "FFFFFF")
        model.members.add(4, 8, "Continuous", 1, 0, "FFFFFF")



        ao = skyciv.ApiObject()
        ao.auth.username = SKYCIV_API_USERNAME  
        ao.auth.key = SKYCIV_API_KEY
        ao.functions.add("S3D.session.start")
        ao.functions.add("S3D.model.set", {"s3d_model": model})
        ao.functions.add("S3D.file.save", {"name": "package-debut", "path": "api/PIP/"})
        res = ao.request()
        output_url = res["response"]["data"]
         
        webbrowser.open_new_tab(output_url)
        
        return output_url  # Return the URL of the SkyCiv model
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')


chat_history = []
@app.route('/ask', methods=['POST'])
def ask():
    global z_coord, y_coord, waiting_for_coordinates, chat_history

    user_input = request.form['user_input']

    if waiting_for_coordinates:
        # User is expected to provide Y-coordinate
        if y_coord is None:
            try:
                y_coord = float(user_input)
                chat_history.append(f"You: {user_input}")
                chat_history.append("Bot: Please enter Z-coordinate.")
                return render_template('index.html', chat_messages=chat_history)
            except ValueError:
                chat_history.append(f"You: {user_input}")
                chat_history.append("Bot: Please enter a valid numeric value for Y-coordinate.")
                return render_template('index.html', chat_messages=chat_history)

        # User is expected to provide Z-coordinate
        elif z_coord is None:
            try:
                z_coord = float(user_input)
                # Both Y and Z coordinates are collected, call SkyCiv API here
                output_url = create_skyciv_model(z_coord, y_coord)
                waiting_for_coordinates = False
                chat_history.append(f"You: {user_input}")
                chat_history.append(f"Bot: SkyCiv model created successfully. You can continue chatting. [Model URL: {output_url}]")
                return render_template('index.html', chat_messages=chat_history)
            except ValueError:
                chat_history.append(f"You: {user_input}")
                chat_history.append("Bot: Please enter a valid numeric value for Z-coordinate.")
                return render_template('index.html', chat_messages=chat_history)

    # Regular chatbot interaction
    if user_input.lower() == 'skyciv':
        waiting_for_coordinates = True
        y_coord = None
        z_coord = None
        chat_history.append(f"You: {user_input}")
        chat_history.append("Bot: Please enter Y-coordinate.")
        return render_template('index.html', chat_messages=chat_history)
    else:
        bot_response = ask_bot(user_input)
        chat_history.append(f"You: {user_input}")
        chat_history.append(f"Bot: {bot_response}")  # Add bot's response to chat history
        return render_template('index.html', chat_messages=chat_history)


@app.route('/skyciv', methods=['POST'])
def skyciv_endpoint():
    global z_coord, y_coord, waiting_for_coordinates
    data = request.json
    z_coord = data.get('z_coord')
    y_coord = data.get('y_coord')

    if z_coord is None or y_coord is None:
        return jsonify({"error": "Both Z-coordinate and Y-coordinate are required."}), 400

    output_url = create_skyciv_model(z_coord, y_coord)
    return jsonify({"message": "SkyCiv model created successfully.", "output_url": output_url})

if __name__ == '__main__':
    app.run(debug=True)
