Temporal Reasoning
==================

To do
-----
- [ ]Use spatial model with only x_c, y_c as relations, box normalization for object states
- [ ]Using the model that uses the cooridinates only
- [ ]Getting object IDs
- [ ]Check the model that use image-based dynamic models
- [x]Handle objects out of scenes
- [ ]Replace the infomation in the t+2 step with GT to see what happen
- [ ]check the original setting for QA task
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


