import os

# Basic Model Configuration
max_length_of_caption = 35
captionModelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model", "bi_model.57.h5")
word_to_idx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tokenization", "word_to_idx.pkl")
idx_to_word_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tokenization", "idx_to_word.pkl")
