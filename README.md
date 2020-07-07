# drums_RNN
This is the code of our RNN for the course Neural Networks (AI)

In this code, an RNN is trained to create a new drum rhythm. It is trained and tested with three songs from U2: I Will Follow, Sunday Bloody Sunday and Pride (In The Name of Love). The training of the RNN consists of a number of epochs in which each time a forward pass, the backpropagation, and updating of the weights is done. During the training a cross-validation scheme is used to prevent overfitting.

To run this code, the csv files of the three songs are needed in the same folder: sunday.csv, follow.csv and pride.csv. In the code, the information needed from these csv files are extracted and put into numpy arrays that can be used by the RNN. 

When the code is run, a new csv file (named result.csv) and the corresponding MIDI file (named result.mid) are created. These files contain the new drum rhythm created by the RNN. Besides this, a picture of a graph is created named flexibility.png. In this graph, the testing loss and training loss of each epoch are displayed to show with how many epochs the RNN was trained without overfitting.

Sanne Berends (s3772950)
Sarah de Boer (s3628701)
Aniek Eijpe (s3756645)
Anniek Theuwissen (s3764818)
