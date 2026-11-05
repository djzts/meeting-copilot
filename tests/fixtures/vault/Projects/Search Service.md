# Search Service

The search service uses Elasticsearch 8 with a BM25 ranking and a learning-to-rank re-ranker.

## Latency

p95 query latency is 180 ms after the October cache change. The SLO is 250 ms.

## Roadmap

Next quarter we add vector search and multilingual analyzers.
