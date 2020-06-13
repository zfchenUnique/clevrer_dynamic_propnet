Temporal Reasoning
==================

To do
-----
- [x]more steps for prediction
- [x]better filtering for prediction data
- [ ]predict offset rather than original values
- [ ]handling outlier
- [x]Better loss for relation features and box sequence
- [x]Data normalization for collision features, object features
- [ ]Larger learning rate

Train
-----

    bash train.sh
    bash train_no_edge_superv.sh
    bash train_no_attr.sh
    bash train_interaction_network.sh

Evaluation
----------

    bash eval.sh
    bash eval_no_edge_superv.sh
    bash eval_interaction_network.sh
    bash eval_no_attr.sh


