step 1: create a virtual env

create a data folder and create a handbook folder within the data folder then upload all the markdown files except for the readme.md

get the the documents from : https://github.com/basecamp/handbook/tree/master

enterprise-knowledge-assistant/
├── data/handbook/        the documents
├── rag/
│   ├── __init__.py
│   ├── loader.py         Module 1: files → Documents
│   ├── chunker.py        Module 2: Documents → chunks
│   ├── embeddings.py     Module 3: text → vectors
│   ├── vector_store.py   Module 4: store + search chunks
│   ├── llm.py            Module 5: Groq chat model
│   ├── prompts.py        Module 6: prompt template
│   └── pipeline.py       Module 6: retrieve → prompt → LLM
├── ingest.py             build the DB
├── chat.py               ask questions
├── .env                  API key
├── .gitignore
└── requirements.txt