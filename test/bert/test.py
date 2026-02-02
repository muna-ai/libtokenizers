from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("test/bert/nomic-embed-text-v1.5.json")
encodings = tokenizer.encode_batch([
    "What is the capital of France?",
    "What is TSNE?"
])
print(encodings[0].ids)