# modalityhallucination-hyperspectral

Inspired by https://people.eecs.berkeley.edu/~jhoffman/papers/Hoffman_CVPR16.pdf
Implemented in Keras.

Hallucinating different modalities of the Indian Pines dataset. Each half of datacube is considered one modality.
Original and hallucinated feature matching using L2 distance is used as hallucination loss along with categorical crossentropy for training the hallucinated network.

Results-

First modality
Teacher - 98.99830883309484%
Student - 99.01131780928841%

Second modality
Teacher - 97.65838428515676%
Student - 98.93326395212697%

