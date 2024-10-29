import nltk

# Ensure the required resource is downloaded
nltk.download('punkt_tab')

# Test the word_tokenize function
test_sentence = "Once upon a time, in a faraway land, there was a magical kingdom."

try:
    tokens = nltk.word_tokenize(test_sentence)
    print("Tokenization successful!")
    print("Tokens:", tokens)
except Exception as e:
    print(f"Error during tokenization: {e}")
