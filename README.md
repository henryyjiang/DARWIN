# DARWIN
DARWIN is an agentic evolutionary AI designed for repeated improvements through being prompted to modify its own source code to improve performance. It utilizes a genetic algorithm-like selection criteria, where a set of parents are prompted to modify the training code of one another to ideally create the next generation of more efficient and higher performance models. Currently, the project is only a proof of concept, utilizing OpenAI for backend and nanoGPT by Andrej Karpathy as the example model. Due to the nature of the agent autonomously modifying the contents of the project directory, it is highly recommended to set up a Docker environment. Users are advised to be aware of the safety risks of unregulated model generated code execution. 

In practice, DARWIN would certainly face time and computation bottlenecks if scaled to large models on the scale of GPT-2 and above. However, the agentic nature of the training makes it uniquely suited to volunteer or grid computing, dividing calculation of local subgradients between volunteers and averaging them for the models trained in each generation. If applied to a product it could be offered as a chatbot service as an executable, using a small portion of GPU or CPU compute power to calculate gradients for training.

#### Additional features to be added soon:
- Parallelization
- SWE-bench performance metrics
- Memory storage of past iteration modifications
- Agentic troubleshooting of errors when models fail
- Interactive window for receiving design changes, library imports, dataset changes, and additional scripts requests from the model during training.

## Setup
```
# API keys, add to ~/.bashrc
export OPENAI_API_KEY='...
```

```
# Verify that Docker is properly configured in your environment.
docker build -t my-app -f DOCKERFILE .
```

```
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```
# Clone SWE-bench
cd swe_bench
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
git checkout dc4c087c2b9e4cefebf2e3d201d27e36
pip install -e .
cd ../../
```

## Running Main Controller:
```
python main/main_controller.py
```

## Acknowledgements:
Credit goes to Andrej Karpathy for the amazing nanoGPT framework. Additional inspiration was taken from great works like Darwin Gödel Machine, Jenny Zhang et. al., and EvoAgentX by Wang et. al.

