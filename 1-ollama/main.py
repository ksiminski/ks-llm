
import ollama

options = {
    "tempature": 0.7,     # response temperature
    "max_tokens": 100,    # maximum number of tokens in the response 
    "num_predict" : 150,  # number of predicted tokens in the response
    "top_p" : 0.9,        # probability distribution sampling
    "num_beams" : 5,      # number of beams in beam search (meaning the model will generate several possible responses and choose the best one)
    "num_ctx" : 2048,     # maximum number of tokens in the context (meaning the model will take into account a specific number of tokens from previous messages)
}

response = ollama.chat(
        model='qwen', 
        messages= [
             {
             'role': 'user',
             'content': 'Write a poem about a cat in the style of Shakespeare.',

             },
        ],
        options=options   # configuration parameters for the model
)

print(response['message']['content'])

