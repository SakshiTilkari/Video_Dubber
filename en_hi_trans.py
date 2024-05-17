import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
# Define the path to your saved model using a raw string
model_path = r"D:\VideoDubber\tf_model"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

def tokenize_text(text):
    """Tokenize the input text."""
    return tokenizer([text], return_tensors='tf')

def translate_text(tokenized_input):
    """Generate translation for the tokenized input."""
    output = model.generate(**tokenized_input, max_length=128)
    return output

def decode_translation(output):
    """Decode the generated translation to a readable format."""
    return tokenizer.decode(output[0], skip_special_tokens=True)

def translate(input_text):
    """Complete translation function: tokenizes input, translates, and decodes the output."""
    tokenized = tokenize_text(input_text)
    translated_output = translate_text(tokenized)
    translated_text = decode_translation(translated_output)
    return translated_text


if __name__ == "__main__":
    input_text = "how are you"
    translated_text = translate(input_text)
    print(f"Translated text: {translated_text}")

