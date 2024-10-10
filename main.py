from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, render_template, request, jsonify

# Initialize tokenizer and model (Replaced with a better model from Huggingface)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form['msg']
    input_text = msg
    return get_Chat_response(input_text)

def get_Chat_response(text):
    # Initialize chat history to None for the first message
    chat_history_ids = None

    # Chat loop for 5 steps
    for step in range(5):
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history or start a new conversation
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

        # Generate a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Pretty print the last output tokens from the bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run(debug=True)
