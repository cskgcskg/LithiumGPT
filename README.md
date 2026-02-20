Lithium 🔋
Lithium is a lightweight, highly optimized character-level Generative Pre-trained Transformer (GPT) written entirely in PyTorch.

This project is heavily inspired by the phenomenal educational work of Andrej Karpathy (specifically his microgpt, makemore, and nanoGPT series). It takes the foundational math of his educational models and scales it up, specifically optimizing the architecture to squeeze maximum performance out of Apple Silicon (M-series chips).

If you want to understand how Large Language Models (LLMs) like ChatGPT work under the hood without getting lost in massive codebases, this repository is for you.

📁 Repository Contents
pytorch_gpt.py: The complete engine. This single file contains the tokenizer, the Transformer model architecture, the training loop, and an interactive chat interface.

input.txt: The training dataset. (Currently configured with the classic trading book Reminiscences of a Stock Operator).

🧠 How It Works
Lithium is a Character-Level Language Model. Instead of learning whole words, it learns to read and write one single character at a time.

Tokenization: The script scans input.txt, finds every unique character (letters, numbers, punctuation), and assigns each a number (an integer ID).

Context Window (Memory): It looks at a specific chunk of characters (the block_size) and attempts to guess what the very next character should be.

The Transformer (Brain): The guesses are processed through Multi-Head Attention blocks and FeedForward layers. It uses backpropagation and Calculus to calculate its "Loss" (how wrong its guess was) and adjusts its internal weights to be smarter on the next step.

Inference (Chat): Once training is complete, the script switches to evaluation mode. You type a prompt, and the AI generates a response character-by-character based on the statistical probabilities of the text it studied.

🍎 Optimized for Apple Silicon (Mac Mini M-Series)
This script has been heavily modified from standard PyTorch CPU tutorials to run at blistering speeds on Apple Silicon (M1/M2/M3/M4 chips) using the Mac's unified memory and GPU.

MPS Backend: Automatically detects and utilizes Apple's Metal Performance Shaders (device = 'mps') instead of falling back to the slower CPU.

Flash Attention: Replaces standard Python for loops in the attention heads with PyTorch's F.scaled_dot_product_attention. This pushes the heavy matrix multiplication down to Apple's highly optimized C++ Metal backend.

Bfloat16 Mixed Precision: Utilizes torch.autocast to train the model in 16-bit precision (bfloat16). This doubles training speed and halves memory usage with zero degradation in learning quality.

Direct-to-GPU Data Loading: Takes advantage of the Mac's Unified Memory architecture by loading the entire dataset tensor directly onto the GPU at initialization, completely eliminating CPU-to-GPU data transfer bottlenecks during the training loop.

🛠️ How to Customize and Experiment
You can easily change the behavior, intelligence, and training data of this AI by modifying a few variables at the top of pytorch_gpt.py.

1. Train on Your Own Data

To teach the AI a new subject, simply delete the contents of input.txt and paste in your own text. (Note: Because this is a character-level model that loads into RAM, keep your file size between 1MB and 5MB for best results. Good examples: Shakespeare plays, C++ source code, or specific niche books).

2. Tweak the Hyperparameters

At the top of the script, you will find the Hyperparameters section. You can scale the model up or down based on your hardware:

block_size (Memory): How many characters the AI looks back at to gain context. Increase this (e.g., 128 to 256) for better grammar and longer context, but it will require more compute power.

n_embd (Brain Width): The number of dimensions in the embeddings.

n_layer (Brain Depth): How many Transformer blocks the data passes through.

dropout (Creativity/Anti-Memorization): Set to 0.2 by default. This randomly turns off 20% of the neurons during training to prevent the AI from perfectly memorizing the book word-for-word, forcing it to learn the underlying patterns instead.

🚀 How to Run
Ensure you have PyTorch installed for Mac:

Bash
pip3 install torch
Run the script:

Bash
python3 pytorch_gpt.py
The script will output the training progress and the decreasing Loss value. Once it finishes its training steps, you will be greeted with an interactive prompt to chat with your newly trained model!
