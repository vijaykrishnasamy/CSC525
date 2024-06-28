from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load fine-tuned model and tokenizer
model_path = './fine_tuned_model'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure the pad_token is set
tokenizer.pad_token = tokenizer.eos_token

# Function to generate response
def generate_response(input_text):
    inputs = tokenizer.encode(input_text, return_tensors='pt', padding=True)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Route to serve the chatbot UI
@app.route('/')
def home():
    return render_template('customerServiceChatbot.html')

# Endpoint for chat (allowing both GET and POST methods)
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        input_text = request.json['text']
        response = generate_response(input_text)
        return jsonify({'response': response})
    else:
        # Handle GET request (optional)
        return 'Chatbot endpoint. Send a POST request with {"text": "your message"} to get a response.'

if __name__ == '__main__':
    app.run(debug=True)
