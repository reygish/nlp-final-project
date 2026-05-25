# Experiment Summary

| Experiment | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
| --- | --- | --- | --- | --- |
| NLTK-NBClassifier | 0.966667 | 0.971015 | 0.966667 | 0.966452 |
| TF-IDF | 0.937500 | 0.952499 | 0.937500 | 0.930916 |
| Word2Vec | 0.879167 | 0.886661 | 0.879167 | 0.879479 |
| GloVe | 0.945833 | 0.952073 | 0.945833 | 0.945777 |

## Summary

- Best overall: NLTK-NBClassifier leads across accuracy and weighted F1.
- GloVe is a close second, outperforming TF-IDF and Word2Vec on all metrics.
- TF-IDF is solid but trails GloVe and NLTK-NBClassifier.
- Word2Vec is the weakest performer among the four experiments.
