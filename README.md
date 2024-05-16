# RAG with Llama3 on Groq


### 1. Create a virtual environment

```shell
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
```

### 2. Export your Groq API Key

```shell
export GROQ_API_KEY=***
```

### 3. Use Ollama or OpenAI for embeddings

Since Groq doesnt provide embeddings yet, you can either use Ollama or OpenAI for embeddings.

- To use Ollama for embeddings [Install Ollama](https://github.com/ollama/ollama?tab=readme-ov-file#macos) and run the `nomic-embed-text` model

```shell
ollama run nomic-embed-text
```

- To use OpenAI for embeddings, export your OpenAI API key

```shell
export OPENAI_API_KEY=sk-***
```


### 4. Install libraries

```shell
pip install -r cookbook/llms/groq/rag/requirements.txt
```

### 5. Run PgVector

> Install [docker desktop](https://docs.docker.com/desktop/install/mac-install/) first.

- Run using a helper script

```shell
./cookbook/run_pgvector.sh
```

- OR run using the docker run command

```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16
```

### 6. Run RAG App

```shell
streamlit run cookbook/llms/groq/rag/app.py
```

- 打开 [localhost:8501](http://localhost:8501) 查看您的 RAG 应用程序。
