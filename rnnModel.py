import numpy as np
import csv
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.special import expit, logit
import py_midicsv

# The arrays for the training dataset, the test dataset, the combined dataset
# and the prime dataset are initialized using numpy arrays.
vector_array_u_train = np.empty(shape=(1007, 2, 1))
vector_array_y_train = np.empty(shape=(1006, 2, 1))
vector_array_u_test = np.empty(shape=(407, 2, 1))
vector_array_y_test = np.empty(shape=(406, 2, 1))
vector_array_u_test_train = np.empty(shape=(1414, 2, 1))
vector_array_y_test_train = np.empty(shape=(1413, 2, 1))
vector = np.empty(shape=(2, 1, 1))
vector_array_prime_u = np.empty(shape=(30, 2, 1))
vector_array_prime_y = np.empty(shape=(29, 2, 1))
# The csv file containing the information of the song 'Sunday Bloody Sunday' of U2 is added to the
# train and the combined datasets.
# For each of the 464 timestamps the loudness of two different drums is added to the arrays.
# The prime array is filled with the first 30 vectors of the train dataset.
with open('sunday.csv', newline='') as f:
    reader = csv.reader(f)
    data = [r for r in reader]

    line_count = 0
    loudness1 = 0
    loudness2 = 0
    quarter_notes = int(data[line_count][1]) / 192
    for i in range(0, 464):
        while int(quarter_notes) == i:

            if int(data[line_count][4]) == 38:
                loudness1 = data[line_count][5]
            if int(data[line_count][4]) == 42:
                loudness2 = data[line_count][5]
            line_count += 1

            quarter_notes = int(data[line_count][1]) / 192

        vector = np.array([[int(loudness1) / 127, int(loudness2) / 127]])
        vector_array_u_train[i] = vector.T
        vector_array_u_test_train[i] = vector.T
        if i < 30:
            vector_array_prime_u[i] = vector.T
        loudness1 = 0
        loudness2 = 0

    for j in range(1, len(vector_array_prime_u)):
        vector_array_prime_y[j-1] = vector_array_prime_u[j]

# The csv file containing the information of the song 'I will follow' of U2 is is added to the
# train and the combined datasets.
# For each of the 534 timestamps the loudness of two different drums is added to the arrays.
with open('follow.csv', newline='') as f:
    reader = csv.reader(f)
    data = [r for r in reader]

    line_count = 0
    loudness1 = 0
    loudness2 = 0
    quarter_notes = int(data[line_count][1]) / 120
    for i in range(0, 543):
        while int(quarter_notes) == i:

            if int(data[line_count][4]) == 38:
                loudness1 = data[line_count][5]
            if int(data[line_count][4]) == 42:
                loudness2 = data[line_count][5]
            line_count += 1

            quarter_notes = int(data[line_count][1]) / 120

        vector = np.array([[int(loudness1) / 127, int(loudness2) / 127]])
        vector_array_u_train[464 + i] = vector.T
        vector_array_u_test_train[464 + i] = vector.T
        loudness1 = 0
        loudness2 = 0

    for j in range(1, len(vector_array_u_train)):
        vector_array_y_train[j-1] = vector_array_u_train[j]
        vector_array_y_test_train[j-1] = vector_array_u_test_train[j]

# The csv file containing the information of the song 'pride' of U2 is added to the test and combined datasets.
# For each of the 407 timestamps the loudness of two different drums is added to the arrays.
with open('pride.csv', newline='') as f:
    reader = csv.reader(f)
    data = [r for r in reader]

    line_count = 0
    loudness1 = 0
    loudness2 = 0
    quarter_notes = int(data[line_count][1]) / 480
    for i in range(0, 407):
        while int(quarter_notes) == i:

            if int(data[line_count][4]) == 38:
                loudness1 = data[line_count][5]
            if int(data[line_count][4]) == 42:
                loudness2 = data[line_count][5]
            line_count += 1

            quarter_notes = int(data[line_count][1]) / 480

        vector = np.array([[int(loudness1) / 127, int(loudness2) / 127]])
        vector_array_u_test[i] = vector.T
        vector_array_u_test_train[871 + i] = vector.T

        loudness1 = 0
        loudness2 = 0

    for j in range(1, len(vector_array_u_test)):
        vector_array_y_test[j-1] = vector_array_u_test[j]
        vector_array_y_test_train[871 + j - 1] = vector_array_u_test_train[871 + j]


# A method that plots the testing loss and the training loss in one graph with on the y-axis the loss and
# on the x-axis the number of epochs.
def plot_losses(testingLoss, trainingLoss, epoch):
    t = np.linspace(0.0, epoch, epoch)
    fig, ax = plt.subplots()
    loss, = ax.plot(t, testingLoss, label = "Testing loss")
    Emploss, = ax.plot(t, trainingLoss, label = "Training loss")

    ax.set(xlabel='Epochs', ylabel='Loss', title='Model flexibility', label='')
    ax.legend(['Testing Loss', 'Training Loss'])
    ax.set_ylim([0, 1])
    ax.grid()

    fig.savefig("flexibility.png")
    plt.show()


# A class defining a recurrent neural network (RNN).
class RNN:
    # The constructor of an RNN. The hidden size, vocab size, learning rate, truncated back propagation through time
    # value, minimal clip value, maximal clip value, alpha value used for ridge regression, weight matrices and bias
    # vector are initialized here.
    def __init__(self):

        self.hidden_size = 176
        self.vocab_size = 2
        self.learning_rate = 0.001

        self.bptt_truncate = 10
        self.min_clip_value = -1
        self.max_clip_value = 1
        self.alfa = 5

        self.Winput = np.random.randn(self.hidden_size, self.vocab_size) * np.sqrt(2/self.vocab_size)
        self.W = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(2/self.hidden_size)
        self.Woutput = np.random.randn(self.vocab_size, self.hidden_size) * np.sqrt(2/self.hidden_size)

        self.b = np.ones((self.hidden_size, 1))

    # A method that applies the derivative of the sigmoid function.
    def sigmoidPrime(self, x):
        return x * (1 - x)

    # A method applies the sigmoid function.
    def sigmoid(self, x):
        return expit(x)

    # A method that returns the next hidden state, it applies the function x[n+1] = sigmoid(W*x[n] + Win*u[n+1] + b).
    def state(self, x, u):
        mulu = np.dot(self.Winput, u)
        mulx = np.dot(self.W, x)
        add = np.add(mulu, mulx)
        return np.array(self.sigmoid(np.add(add, self.b)))

    # A method that checks the loss given an input array and an output array representing the expected output with the
    # input array, an output array is computed. This computed output array is then compared to the expected output array
    # to compute the loss. The loss is computed with the mean squared error.
    def checkLoss(self, U, Y):
        # Initialization of x (the hidden states) with zeros:
        x = np.zeros((self.hidden_size, 1))
        # Computation of the output array given the input array:
        YHat, _ = self.forward(U, x, Y)
        # Initialization of the loss vector with zeros:
        loss = 0
        # Computation for each timestamp of the loss (Yhat - Y)^2:
        for i in range(len(Y)):
            yHat, y = YHat[i], Y[i]
            d = abs(yHat-y)
            dot = np.dot(d.T, d)
            loss += dot

        # Averaging of the loss to get the mean squared error:
        risk = loss / len(Y)
        return risk

    # A method that performs the forward pass in the recurrent neural network. The function
    # y[n] = sigmoid(Wout*sigmoid(W*x[n-1] + Win[n] + b)) is applied.
    def forward(self, U, x, Y):
        preds = []
        hidden_states = []
        # For each timestep an output array (yHat) and the hidden states are computed, the output arrays are added to
        # the prediction array and the hidden states are added to the hidden states array for later use in the truncated
        # back propagation through time algorithm.
        for i in range(len(Y)):
            u, y = U[i], Y[i]
            x = self.state(x, u)
            yHat = np.dot(self.Woutput, x)
            yHat = self.sigmoid(yHat)
            hidden_states.append(x)
            preds.append(yHat)
        return np.array(preds), np.array(hidden_states)

    # A method that performs the truncated back propagation through time algorithm.
    # The gradients of the weight matrices are computed with the use of the derivative of the loss.
    def backprop(self, yHat, U, x, Y):
        dWin = np.zeros(self.Winput.shape)
        dWout = np.zeros(self.Woutput.shape)
        dW = np.zeros(self.W.shape)

        for t in range(len(Y))[::-1]:
            # The derivative of the loss is computed.
            delta_loss = np.array(2 * (abs(yHat[t] - Y[t])))
            # The gradient of the Wout matrix is computed.
            dWout += np.outer(delta_loss, x[t])
            # delta_t is the derivative of the loss multiplied with the Wout matrix,
            # and then multiplied with the partial derivative of the activation value.
            # This value is updated in the loop that follows.
            delta_t = np.dot(np.transpose(self.Woutput), delta_loss)
            delta_t = delta_t * self.sigmoidPrime(x[t])

            maximum = max(0, t-self.bptt_truncate)
            # This loop is the backward pass of the algorithm, it loops back till the truncate value.
            for timestep in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # The gradient of the W matrix is the sum of all the delta_t's times the hidden activation value.
                dW += np.outer(delta_t, x[timestep - 1])
                # The gradient of the Win matrix is the sum of all the delta_t's times the input value.
                dWin += np.outer(delta_t, U[timestep])
                # The delta_t is updated each step you go back through the network, now instead of multiplying with
                # the Wout, you multiply with the W matrix.
                delta_t = np.dot(np.transpose(self.W), delta_t)
                delta_t = delta_t * self.sigmoidPrime(x[timestep-1])

            # Average
            dW /= (t+1-maximum)
            dWin /= (t+1-maximum)

        # Average over the length.
        dWin /= len(Y)
        dW /= len(Y)
        dWout /= len(Y)
        return dWin, dW, dWout

    # A method that updates the weight matrices.
    def updateWeights(self, dWin, dW, dWout, Y):
        # Preventing from exploding gradient problem:
        if dWin.max() > self.max_clip_value:
            dWin[dWin > self.max_clip_value] = self.max_clip_value
        if dW.max() > self.max_clip_value:
            dW[dW > self.max_clip_value] = self.max_clip_value
        if dWout.max() > self.max_clip_value:
            dWout[dWout > self.max_clip_value] = self.max_clip_value

        if dWin.min() < self.min_clip_value:
            dWin[dWin < self.min_clip_value] = self.min_clip_value
        if dW.min() < self.min_clip_value:
            dW[dW < self.min_clip_value] = self.min_clip_value
        if dWout.min() < self.min_clip_value:
            dWout[dWout < self.min_clip_value] = self.min_clip_value

        # Updates the weights with the gradients and applies the ridge regression:
        self.Winput = self.Winput - (self.learning_rate * dWin + 2 * self.alfa * (self.Winput / len(Y)))
        self.W = self.W - (self.learning_rate * dW + 2 * self.alfa * (self.W / len(Y)))
        self.Woutput = self.Woutput - (self.learning_rate * dWout + 2 * self.alfa * (self.Woutput / len(Y)))

    # A method that trains the recurrent neural network by performing the forward pass and the truncated backpropagation
    # through time and that updates the weight matrices.
    def training(self, U, Y):
        x = np.zeros((self.hidden_size, 1))
        yHat, hidden_states = self.forward(U, x, Y)
        dWin, dW, dWout = self.backprop(yHat, U, hidden_states, Y)
        self.updateWeights(dWin, dW, dWout, Y)

    # A method that computes an output array with a prime array.
    def prediction(self, U, Y):
        hidden_states = []
        preds = []
        prev_x = np.zeros((self.hidden_size, 1))
        #  Computation of the outputs with the use of the prime array, the same way as in the forward function.
        for i in range(len(Y)):
            u = U[i]
            x = self.state(prev_x, u)
            yHat = np.dot(self.Woutput, x)
            yHat = self.sigmoid(yHat)
            hidden_states.append(x)
            prev_x = x
        # Computation of the outputs in which the previous output is used as the new input.
        prev_y = yHat
        for j in range(len(U), 100):
            x = self.state(prev_x, prev_y)
            out = np.dot(self.Woutput, x)
            yHat = self.sigmoid(out)
            hidden_states.append(x)
            preds.append(yHat)
            prev_x = x
            prev_y = yHat
        return np.array(preds)


# A recurrent neural network is initialized and copied to be used for cross validation.
rnn = RNN()
rnnRun = deepcopy(rnn)

# A loop to compute the number of epochs that prevents from overfitting. This number of epochs is computed by increasing
# the number until the testing loss no longer decreases.
epoch = 0
previous_Testloss = 10000
prev_loss = 10000
testLoss = rnn.checkLoss(vector_array_u_test, vector_array_y_test)
trainLoss = rnn.checkLoss(vector_array_u_train, vector_array_y_train)

testLosses = []
trainingLosses = []
trainLossDiminish = 0
while previous_Testloss >= testLoss and trainLossDiminish != 5:
    if(prev_loss < trainLoss):
        trainLossDiminish += 1
    epoch += 1
    trainingLosses.append(trainLoss)
    testLosses.append(testLoss)
    prev_loss = trainLoss
    previous_Testloss = testLoss
    rnn.training(vector_array_u_train, vector_array_y_train)
    trainLoss = rnn.checkLoss(vector_array_u_train, vector_array_y_train)
    testLoss = rnn.checkLoss(vector_array_u_test, vector_array_y_test)
    print('Epoch: ', epoch, ', Loss: ', trainLoss, ', Val Loss: ', testLoss)

n = epoch

# The training is run for some more epochs to show in the model flexibility plot.
for i in range(10):
    n += 1
    trainingLosses.append(trainLoss)
    testLosses.append(testLoss)
    prev_loss = trainLoss
    previous_Testloss = testLoss
    rnn.training(vector_array_u_train, vector_array_y_train)
    trainLoss = rnn.checkLoss(vector_array_u_train, vector_array_y_train)
    testLoss = rnn.checkLoss(vector_array_u_test, vector_array_y_test)

testLosses = np.reshape(testLosses, (len(testLosses)))
trainingLosses = np.reshape(trainingLosses, (len(trainingLosses)))
plot_losses(testLosses, trainingLosses, n)


# The actual training of the recurrent neural network with the right number of epochs.
for i in range(epoch):
    rnnRun.training(vector_array_u_test_train, vector_array_y_test_train)
    loss = rnnRun.checkLoss(vector_array_u_test_train, vector_array_y_test_train)
    print('Epoch: ', i, ', Loss: ', loss)

# The computation of the output with the use of the prime array:
result = rnnRun.prediction(vector_array_prime_u, vector_array_prime_y)

for i in range(0, len(result)):
    y = result[i]
    loudness1 = int(y[0] * 127)
    loudness2 = int(y[1] * 127)
    print(loudness1, loudness2)

# The output matrix is transformed into a csv file.
with open('result.csv', 'w') as f:
    f.write('0, 0, Header, 1, 2, 264\n')
    f.write('1, 0, Start_track\n')
    f.write('1, 0, Key_signature, 0, "major"\n')
    f.write('1, 0, Tempo, 501133\n')
    f.write('1, 0, Time_signature, 4, 2, 24, 8\n')
    f.write('1, 0, End_track\n')
    f.write('2, 0, Start_track\n')
    f.write('2, 0, Title_t, "drums"\n')
    f.write('2, 0, MIDI_port, 0\n')
    for i in range(0, len(result)):
        y = result[i]
        midi_clock = i * 264

        loudness1 = int(y[0] * 127)
        loudness2 = int(y[1] * 127)
        f.write('2, {}, Note_on_c, 9, 38, {}\n'.format(midi_clock, loudness1))
        f.write('2, {}, Note_on_c, 9, 42, {}\n'.format(midi_clock, loudness2))
    f.write('2, 0, End_track\n')
    f.write('0, 0, End_of_file\n')


midi = py_midicsv.csv_to_midi('result.csv')

# the csv file containing the information of the output is transformed into a midi file.
with open("result.mid", "wb") as output_file:
    midi_writer = py_midicsv.FileWriter(output_file)
    midi_writer.write(midi)

