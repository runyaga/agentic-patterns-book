# Production Reality Check: Domain Exploration (The Cartographer)

**Target file**: `docs/patterns/21-domain-exploration.md`
**Replaces**: `## Production Reality Check` section (expand existing)

---

## Production Reality Check

### When to Use
- Need autonomous discovery of unknown domain (new codebase, documentation set)
- Want semantic understanding beyond keyword search (entity relationships)
- Building knowledge graphs for structural queries (dependencies, call graphs)
- Gap detection is valuable (find undocumented functions, orphan modules)

### When NOT to Use
- Corpus is small and can be indexed manually
- Simple keyword search meets your needs
- Real-time exploration isn't acceptable (crawling is slow)
- Domain is well-understood and doesn't need discovery

**Key insight**: Most codebases and document sets can be understood with simpler
tools (grep, ctags, IDE indexers). Use Cartographer when semantic relationships
matter more than text matches.

### Production Considerations
- **Boundaries**: Set strict `max_depth` and `max_files` limits. Unbounded
  crawling is expensive and can loop infinitely on large repos.
- **Memory management**: Large knowledge graphs should persist to disk
  (JSON/LanceDB) periodically. Don't hold everything in RAM.
- **AST-first accuracy**: LLMs hallucinate relationships. For code, use AST
  (Abstract Syntax Tree) for ground truth (imports, class hierarchy). Use LLM
  only for semantic summaries and conceptual relationships.
- **Incremental updates**: Full re-crawl is expensive. Implement incremental
  indexing for changed files only.
- **Token budgeting**: Each file processed = LLM call for entity extraction.
  Provide "dry run" mode to preview what would be crawled before committing.
- **Entity ID stability**: Content-based hashing for entity IDs (not path-based)
  prevents duplicates when files move.
- **Visualization**: Knowledge graphs need visualization to be useful. Plan for
  export to GraphViz, Neo4j, or similar tools.
