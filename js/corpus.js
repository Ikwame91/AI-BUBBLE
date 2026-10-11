// Training text for the in-browser toy models, and the knowledge base used by the RAG demo.
// Everything the "models" on this site know comes from these strings — nothing else.
window.LLM = window.LLM || {};

LLM.CORPUS = `the cat sat on the mat. the cat was happy on the mat. the dog sat on the rug. the dog was happy on the rug.
the cat saw the dog. the dog saw the cat. the cat ran to the door and the dog ran after the cat.
the little cat liked warm milk. the big dog liked a long walk in the park. the cat and the dog were friends.
one morning the sun was warm and the cat sat in the sun. the dog sat in the shade under the old tree.
the girl gave the dog a bone. the boy gave the cat some milk. the girl and the boy played in the park.
the king lived in a large castle. the queen lived in the castle with the king. the king and the queen ruled the land.
the prince rode a white horse to the castle. the princess read a book in the tower of the castle.
the model reads the text one token at a time. the model predicts the next token. then the model reads that token and predicts the next one.
a language model is a machine that predicts the next word. it learns from a large amount of text.
the model learns patterns in the text. it does not look up answers in a table. it computes a probability for every token.
each token becomes a vector of numbers. the vector is called an embedding. words with similar meaning get similar vectors.
attention lets each token look at the other tokens. a token can attend to the words that matter for it.
the transformer is made of many layers. each layer has attention and a small neural network.
training changes the weights of the network so the next token gets a higher probability.
the loss measures how surprised the model is. when the loss goes down the model is learning.
the model is trained on books and websites and code. after training the model can write text.
a chat model is a language model that was trained again to follow instructions and to be helpful.
people rate the answers of the model and the model learns to give better answers.
a reasoning model writes its thinking before it gives the final answer. more thinking can lead to better answers.
the robot read the book about the stars. the stars were bright in the dark sky. the moon was full and white.
the old man walked to the market. he bought bread and apples and cheese. the bread was warm and the apples were red.
the farmer worked in the field all day. the field was green after the rain. the rain fell on the field and on the farm.
in the evening the family ate dinner together. they talked about the day and they laughed.
the teacher wrote on the board. the students read the words on the board and they learned.
the river ran past the town. the children played by the river in the summer. in the winter the river was cold.
the cat sat by the window and watched the rain. the dog slept by the fire. the house was quiet and warm.
the ship sailed across the sea. the sailors saw a small island. on the island there was a tall tree and a quiet beach.
the bird sang in the tree. the bird flew over the river and over the town and over the field.
the computer runs the program. the program reads the data and writes the answer. the answer was correct.
the king gave the prince a sword. the queen gave the princess a crown. the prince and the princess were brave.
the man and the woman walked in the park. the woman read a book and the man fed the birds.
the model does not know what is true. it knows what is likely. that is why it can make mistakes.
a good answer is clear and correct and kind. a bad answer is confusing or wrong.
the cat sat on the mat again. the dog sat on the rug again. and the story ended there.`;

// Short documents for the retrieval (RAG) demo. Each one is a chunk a real system would store in a vector database.
LLM.DOCS = [
  { id: "tokens", title: "Tokens", text: "A token is the unit of text a language model reads and writes. Tokenizers such as byte-pair encoding split text into frequent pieces: common words become one token, rare words split into several. English averages roughly four characters per token. Models count context length and price in tokens." },
  { id: "embeddings", title: "Embeddings", text: "An embedding is a vector of numbers that represents a token. The model looks up each token id in an embedding table. Tokens used in similar contexts end up with similar vectors, so distance between vectors reflects similarity of meaning." },
  { id: "attention", title: "Self-attention", text: "Self-attention lets every token gather information from earlier tokens. Each token produces a query, a key and a value vector. The dot product of a query with each key, scaled and passed through softmax, gives attention weights that mix the value vectors. A causal mask stops tokens from seeing the future." },
  { id: "transformer", title: "The transformer", text: "The transformer architecture was introduced in 2017 in the paper Attention Is All You Need. A transformer stacks many identical blocks. Each block has multi-head self-attention followed by a feed-forward network, with residual connections and layer normalization. GPT, Claude, Llama, Gemini and DeepSeek models are all transformer decoders." },
  { id: "sampling", title: "Sampling and temperature", text: "At each step the model outputs logits, one score per vocabulary token. Softmax turns logits into probabilities. Temperature divides the logits before softmax: low temperature makes output predictable, high temperature makes it varied. Top-k keeps the k most likely tokens; top-p keeps the smallest set whose probability adds up to p." },
  { id: "pretraining", title: "Pretraining", text: "Pretraining teaches a model to predict the next token on trillions of tokens of text from the web, books and code. The objective is cross-entropy loss. Pretraining produces a base model that continues text but does not follow instructions reliably." },
  { id: "rlhf", title: "RLHF and alignment", text: "Reinforcement learning from human feedback turns a base model into an assistant. First, supervised fine-tuning on example conversations. Then people compare pairs of answers, a reward model learns their preferences, and the language model is optimized against that reward. Constitutional AI replaces many human labels with feedback from a model guided by written principles. DPO optimizes on preference pairs directly without a separate reward model." },
  { id: "reasoning", title: "Reasoning models", text: "Reasoning models such as OpenAI o1 and DeepSeek-R1 are trained with reinforcement learning to write a long chain of thought before answering. Rewards come from checkable outcomes such as correct math answers or passing unit tests. Spending more tokens on thinking at inference time, called test-time compute, improves accuracy on hard problems." },
  { id: "rag", title: "Retrieval-augmented generation", text: "Retrieval-augmented generation (RAG) fetches relevant documents and pastes them into the prompt before the model answers. Search-based assistants like Perplexity run a web search, extract relevant passages, rank them, and ask the model to answer with citations. RAG reduces hallucination and lets a model use information newer than its training data." },
  { id: "agents", title: "Tools and agents", text: "Tool use lets a model call functions. The application describes tools with names and JSON schemas. The model replies with a tool call, the application runs it and sends back the result, and the loop repeats until the model gives a final answer. An agent is this loop running over many steps. The Model Context Protocol (MCP) is a standard way to connect tools and data to models." },
  { id: "context", title: "Context window", text: "The context window is the maximum number of tokens a model can read at once, including the system prompt, the conversation history, retrieved documents and its own output. A chat model has no memory between requests: the application resends the whole conversation every turn." },
  { id: "hallucination", title: "Hallucination", text: "A hallucination is a fluent answer that is false. It happens because the model generates likely text, not verified facts. Retrieval, citations, tool use, and training the model to say when it is unsure all reduce hallucination." },
  { id: "moe", title: "Mixture of experts", text: "A mixture-of-experts model has many feed-forward experts per layer and a router that sends each token to only a few of them. The model has many parameters but uses a fraction per token, which makes it cheaper to run. Mixtral and DeepSeek-V3 are mixture-of-experts models." },
  { id: "interpretability", title: "Interpretability", text: "Mechanistic interpretability tries to read the internal computations of a neural network. Sparse autoencoders find features, directions in activation space that match concepts. Anthropic's circuit tracing builds attribution graphs that show, for example, Claude first representing Texas internally before answering that the capital of the state containing Dallas is Austin." },
  { id: "scaling", title: "Scaling laws", text: "Scaling laws describe how loss falls predictably as you increase parameters, data and compute. The Chinchilla result showed models should be trained on about twenty tokens per parameter for compute-optimal training. Test-time compute is a second scaling axis: letting the model think longer." }
];
