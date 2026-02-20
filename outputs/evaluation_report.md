
# RAG System Evaluation Report
Generated: February 19, 2026 at 07:21 PM

## Executive Summary
- **Overall Accuracy:** 53.3%
- **Total Questions:** 15
- **Correct Answers:** 6
- **Partial Answers:** 4
- **Incorrect Answers:** 5

## Performance by Category

### Factual Questions (Single Lookup)
- Accuracy: 65.0%
- Count: 10
- Strength: Best performing category

### Comparative Questions (Cross-Document)
- Accuracy: 33.3%
- Count: 3
- Challenge: Requires synthesis across companies

### Trend Questions (Multi-Year)
- Accuracy: 25.0%
- Count: 2
- Challenge: Needs data from multiple years

## Common Failure Modes

1. **Numerical Precision** (3 questions affected)
   - Different units ($M vs $B vs $T)
   - Rounding differences
   - Missing decimal places

2. **Cross-Document Synthesis** (2 questions affected)
   - Comparing data across banks
   - Requires multiple retrievals
   - Context prioritizes one company

3. **Multi-Year Trends** (2 questions affected)
   - Needs 2-3 years of data
   - Chunks may not span all years
   - Temporal reasoning required

4. **Missing Context** (2 questions affected)
   - Information split across chunks
   - Key details in different sections
   - Chunking breaks up complete info

## Recommendations

### Short-term Improvements
- Adjust k value based on question type
- Add numerical normalization post-processing
- Implement better error messages for missing data

### Long-term Enhancements
- Query routing (factual vs comparative)
- Hybrid retrieval (dense + sparse)
- Fine-tune embeddings on financial documents
- Larger context window models (GPT-4)

## Conclusion
The RAG system performs well on factual lookups (60% accuracy on factual questions) 
but struggles with cross-document comparisons and multi-year trends (25% on comparative, 
25% on trend questions). With targeted improvements, accuracy could reach 75-80%.
