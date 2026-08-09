Dissertation Research Plan: Enhancing Domain-Specific Document Understanding in UK Planning Policy using Advanced RAG Architectures

1. Project Overview & Objectives

Goal: To evaluate how different embedding and retrieval strategies within Hierarchical and Graph RAG architectures enrich the understanding of domain-specific, structurally complex texts (UK Planning Policies) and reduce LLM hallucinations.
Dataset: 1.50 GB total. Focusing strictly on:

General UK Planning Policy (dataset/general).

Coventry Local Plan (dataset/Local Statuatory Development Plans/Coventry).

2. The Core Tech Stack (Software & Algorithmic Focus)

Orchestration: LlamaIndex

Vector Store: Milvus or Qdrant (Scalable, disk-backed vector storage).

Graph Store: Neo4j (Cypher-based querying for multi-hop policy relationships).

Serving: vLLM serving Llama-4-Maverick-17B-128E-Instruct-FP8.

Embeddings: BAAI/bge-large-en-v1.5, nlpaueb/legal-bert-base-uncased, and a Custom Fine-Tuned Model.

Evaluation: RAGAS (with GPT-4.1 serving as the evaluator/judge).

3. Advanced Data Preprocessing Pipeline

Handling a 1.5 GB dataset requires programmatic data cleaning, deduplication, and intelligent metadata tagging. "Garbage in, garbage out" is especially true for legal documents.

3.1 Deep-Directory Path Tagging (Document Classification)

Crucial Rule: Do not manually merge or flatten the sub-sub-folders. The deep directory structure contains highly valuable context.

Mechanism: Use LlamaIndex's SimpleDirectoryReader(input_dir="...", recursive=True). The recursive=True flag automatically crawls through all deep sub-sub-folders without requiring you to move any files.

Dynamic Path Parsing: We will pass a custom file_metadata function to extract both the top-level category and the specific nested sub-folder context.

Logic:

If filepath contains "1. Adopted Coventry Local Plan", tag as {"document_class": "Statutory Policy", "region": "Coventry"}.

The script will also split the filepath string to extract the immediate parent folder name (e.g., if the file is in .../2. Supplementary Planning Documents/Affordable Housing/doc.pdf, it automatically adds the tag {"sub_topic": "Affordable Housing"}).

Impact: Preserves the exact original structure of the legal documents while allowing the retriever to filter by highly specific sub-topics, preventing the loss of deep-folder context.

3.2 LLM-Powered Metadata Extraction

Standard parsing is insufficient for complex planning jargon. We will use the Llama-4-Maverick model during the ingestion phase to read each chunk and extract semantic tags.

Mechanism: Use LlamaIndex's PydanticProgramExtractor or SummaryExtractor.

Implementation: As the pipeline processes chunks, the LLM is prompted to output a JSON schema identifying specific entities (e.g., ["Green Belt", "Affordable Housing"]) and a concise summary. This metadata is injected directly into the chunk's vector payload.

3.3 Document Deduplication

Redundant policy documents inflate storage costs and heavily skew vector retrieval towards duplicate chunks.

Mechanism: Attach a SimpleDocumentStore to the LlamaIndex IngestionPipeline and set the docstore_strategy=DocstoreStrategy.UPSERTS.

Execution: The pipeline creates a unique hash for every document. If a document hash already exists in the document store, the pipeline skips it, ensuring 0% duplication across your 1.5 GB dataset.

3.4 Hierarchical Node Parsing

Chunk Sizes: Create a hierarchy: [2048, 512, 128].

Execution: 128-token leaf chunks are embedded. When matched by a query, the system retrieves the parent 2048-token chunk to preserve the full policy context.

4. The Step-by-Step Experimental Progression

To scientifically prove your findings, you must establish a baseline and change variables logically.

Phase 0: Data Preprocessing & Ground Truth Generation

Action: Run the 1.5GB dataset through the IngestionPipeline (Recursive Path Tagging -> LLM Extraction -> Deduplication -> Hierarchical Chunking).

Action: Use GPT-4.1 to generate a "Golden Dataset" of 50-100 evaluation questions and answers, testing both policy rules and contextual evidence.

Step 1: The Baseline (Naive RAG)

Configuration: Standard Chunking (512 tokens), General Embedding (bge-large), Pure Vector Search.

Hypothesis: It will struggle with specific policy codes and confuse guidance with strict policy.

Step 2: Upgrading Retrieval Strategy

Configuration: Add BM25 (Hybrid Search) and a Cross-Encoder (Jina AI Reranker).

Hypothesis: Sharp increase in RAGAS Context Precision due to exact keyword matching (e.g., "Policy DS3").

Step 3: Upgrading the Architecture (Hierarchical RAG)

Configuration: Implement AutoMergingRetriever using the Hierarchical chunks created in Phase 0.

Hypothesis: Major improvement in RAGAS Answer Relevance and Faithfulness as parent context is restored.

Step 4: Upgrading Domain Knowledge (Embeddings)

Configuration: Swap the general embedding model for LEGAL-BERT, and subsequently, a Custom Fine-Tuned Embedding.

Hypothesis: Fine-tuning the embeddings to specifically understand Coventry planning jargon will yield the highest vector retrieval scores.

Step 5: Tackling Multi-Hop Complexity (Graph RAG)

Configuration: Introduce PropertyGraphIndex for the Coventry dataset to map entity relationships. Combine with Hierarchical RAG.

Hypothesis: Achieves highest RAGAS Context Recall for complex, cross-referencing user queries.

5. Evaluation Framework (RAGAS)

Run your pipelines using Llama-4-Maverick, evaluating with GPT-4.1 via RAGAS for:

Faithfulness: Measures if the claims in the generated answer can be inferred directly from the retrieved context (Hallucination check).

Answer Relevance: Measures how well the answer addresses the actual question.

Context Precision: Measures whether the relevant chunks were ranked highest.

Context Recall: Measures if all necessary information to answer the question was successfully retrieved.