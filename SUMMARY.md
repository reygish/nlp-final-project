# Experiment Summary

| Experiment | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
| --- | --- | --- | --- | --- |
| Bag-of-Words | 0.966667 | 0.969820 | 0.966667 | 0.966416 |
| TF-IDF | 0.970833 | 0.972889 | 0.970833 | 0.970814 |
| Word2Vec | 0.879167 | 0.886661 | 0.879167 | 0.879479 |
| GloVe | 0.945833 | 0.952073 | 0.945833 | 0.945777 |

## Summary

- Best overall: TF-IDF leads across accuracy and weighted F1.
- Bag-of-Words is a close second, outperforming GloVe and Word2Vec on all metrics.
- Word2Vec is the weakest performer among the four experiments.
