import numpy as np
import csv
import matplotlib.pyplot as plt
import py_midicsv

# The csv file containing the information of the song 'sunday' of U2 is transformed to a 464 by 2 matrix,
# representing for each of the 464 timestamps the drum hits of two different drums. This matrix is used as
# training data.
vector_array_u_train = []
vector_array_y_train = []
with open('sunday.csv', newline='') as f:
    reader = csv.reader(f)
    data = [r for r in reader]

    line_count = 0
    loudness1 = 0
    loudness2 = 0
    quarter_notes = int(data[line_count][1]) / 192
    sixteenth_notes = quarter_notes
    for i in range(0, 464):
        while int(sixteenth_notes) == i:

            if int(data[line_count][4]) == 38:
                loudness1 = data[line_count][5]
            if int(data[line_count][4]) == 42:
                loudness2 = data[line_count][5]
            line_count += 1

            quarter_notes = int(data[line_count][1]) / 192
            sixteenth_notes = quarter_notes

        vector = [float(loudness1)/127, float(loudness2)/127]
        print(vector)
        vector_array_u_train.append(vector)
        loudness1 = 0
        loudness2 = 0

# The csv file containing the information of the song 'follow' of U2 is transformed to a 543 by 2 matrix,
# representing for each of the 543 timestamps the drum hits of two different drums. This matrix is also
# used as training data.
with open('follow.csv', newline='') as f:
    reader = csv.reader(f)
    data = [r for r in reader]

    line_count = 0
    loudness1 = 0
    loudness2 = 0
    quarter_notes = int(data[line_count][1]) / 120
    sixteenth_notes = quarter_notes
    for i in range(0, 543):
        while int(sixteenth_notes) == i:

            if int(data[line_count][4]) == 38:
                loudness1 = data[line_count][5]
            if int(data[line_count][4]) == 42:
                loudness2 = data[line_count][5]
            line_count += 1

            quarter_notes = int(data[line_count][1]) / 120
            sixteenth_notes = quarter_notes

        vector = [int(loudness1)/127, int(loudness2)/127]
        print(vector)
        vector_array_u_train.append(vector)
        loudness1 = 0
        loudness2 = 0

    for j in range(1, len(vector_array_u_train)):
        vector_array_y_train.append(vector_array_u_train[j])

    vector_array_u_train = np.array(vector_array_u_train)
    vector_array_y_train = np.array(vector_array_y_train)

vector_array_u_test = []
vector_array_y_test = []

# The csv file containing the information of the song 'pride' of U2 is transformed to a  407 by 2 matrix,
# representing for each of the 407 timestamps the drum hits of two different drums. This matrix is used as
# testing data.
with open('pride.csv', newline='') as f:
    reader = csv.reader(f)
    data = [r for r in reader]

    line_count = 0
    loudness1 = 0
    loudness2 = 0
    quarter_notes = int(data[line_count][1]) / 480
    sixteenth_notes = quarter_notes
    for i in range(0, 407):
        while int(sixteenth_notes) == i:

            if int(data[line_count][4]) == 38:
                loudness1 = data[line_count][5]
            if int(data[line_count][4]) == 42:
                loudness2 = data[line_count][5]
            line_count += 1

            quarter_notes = int(data[line_count][1]) / 480 # verandert per liedje
            sixteenth_notes = quarter_notes

        vector = [int(loudness1)/127, int(loudness2)/127]
        print(vector)
        vector_array_u_test.append(vector)

        loudness1 = 0
        loudness2 = 0

    for j in range(1, len(vector_array_u_test)):
        vector_array_y_test.append(vector_array_u_test[j])

    vector_array_u_test = np.array(vector_array_u_test)
    vector_array_y_test = np.array(vector_array_y_test)

#https://github.com/nikhilroxtomar/Binary-to-Decimal-using-Recurrent-Neural-Network/blob/master/bin2int-rnn.py
#https://www.analyticsvidhya.com/blog/2019/01/fundamentals-deep-learning-recurrent-neural-networks-scratch-python/
#http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
#https://d2l.ai/chapter_recurrent-neural-networks/bptt.html
#https://towardsdatascience.com/understanding-the-scaling-of-l%C2%B2-regularization-in-the-context-of-neural-networks-e3d25f8b50db
#https://missinglink.ai/guides/neural-network-concepts/neural-networks-regression-part-1-overkill-opportunity/

# A method that plots the testing loss and the training loss in one graph with on the y-axis the loss and on the x-axis the
# number of epochs.
def plot_losses(testingLoss, trainingLoss, epoch):
    t = np.linspace(0.0, epoch, epoch)
    fig, ax = plt.subplots()
    ax.plot(t, testingLoss, label = "loss")
    ax.plot(t, trainingLoss, label = "Emperical loss")

    ax.set(xlabel='Epochs', ylabel='Loss/emperical risk', title='Model flexibility', label='')
    ax.grid()

    fig.savefig("flexibility.png")
    plt.show()

# A class defining a recurrent neural network (RNN).
class RNN:
    # The constructor of an RNN. The hidden size, vocab size, learning rate, truncated back propagation through time
    # value, minimal clip value, maximal clip value, alpha value used for ridge regression, weight matrices and bias
    # vector are initialized here.
    # The hidden size is initialized with 500 EXPLAIN WHY. The vocab size is initialized with 2 because each timestamp
    # contains two pieces of information: the drum hit of two different drums. The learning rate is initialized with
    # 0.001 EXPLAIN WHY. The truncated back propagation through time value is initialized with 5 EXPLAIN WHY. The minimal
    # clip value is initialized with -1 EXPLAIN WHY. The maximal clip value is initialized with 1 EXPLAIN WHY. The alpha
    # value used for ridge regression is initialized with 2 EXPLAIN WHY. The weight matrices are initialized with random
    # values drawn from a uniform distribution of the interval [-sqrt(1/2), sqrt(1/2)] EXPLAiN WHY. The bias vector is
    # initialized with ones EXPLAIN WHY.
    def __init__(self):

        self.hidden_size = 500
        self.vocab_size = 2
        self.learning_rate = 0.001

        self.bptt_truncate = 5
        self.min_clip_value = -1
        self.max_clip_value = 1
        self.alfa = 2

        self.Winput = np.random.uniform(-np.sqrt(1./self.vocab_size), np.sqrt(1./self.vocab_size), (self.hidden_size, self.vocab_size))
        self.W = np.random.uniform(-np.sqrt(1./self.vocab_size), np.sqrt(1./self.vocab_size), (self.hidden_size, self.hidden_size))
        self.Woutput = np.random.uniform(-np.sqrt(1./self.vocab_size), np.sqrt(1./self.vocab_size), (self.vocab_size, self.hidden_size))

        self.b = np.ones((self.hidden_size, 1))

    # A method that applies the derivative of the sigmoid function.
    def sigmoidPrime(self, x):
        y = self.sigmoid(x)
        return y*(1-y)

    # A method applies the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # A method that returns the next hidden state, it applies the function x[n+1] = sigmoid(W*x[n] + Win*u[n+1] + b).
    def state(self, x, u):
        u = np.reshape(u, (2,1))
        # Win*u[n+1]:
        mulu = np.dot(self.Winput, u)
        # W*x[n]:
        mulx = np.dot(self.W, x)
        # W*x[n] + Win*u[n+1]:
        add = np.add(mulu, mulx)
        #sigmoid(W*x[n] + Win*u[n+1] + b):
        lol = self.sigmoid(np.add(add, self.b))
        return np.array(lol)

    # A method that checks the loss given an input array and an output array representing the expected output With the
    # input array, an output array is computed. This computed output array is then compared to the expected output array
    # to compute the loss. The loss is computed with the mean squared error.
    def checkLoss(self, U, Y):
        # initialization of x (the hidden states) with zeros:
        x = np.zeros((self.hidden_size, 1))
        # computation of the output array given the input array:
        YHat, _ = self.forward(U, x, Y)
        # initialization of the loss vector with zeros:
        loss = np.zeros((self.vocab_size, 1))
        # computation for each timestamp of the loss (Yhat - Y)^2, this is the squared error:
        for i in range(len(Y)):
            yHat, y = YHat[i], Y[i]
            yHat = np.reshape(yHat, (2, 1))
            y = np.reshape(y, (2, 1))

            loss += (abs(y-yHat))**2
        # averaging of the loss to get the mean squared error:
        risk = loss / len(Y)
        return risk

    # A method that performs the forward pass in the recurrent neural network. The function
    # y[n] = sigmoid(Wout*sigmoid(W*x[n-1]+Win[n] + b)) is applied.
    def forward(self, U, x, Y):
        preds = []
        hidden_states = []
        # for each timestamp an output array (yHat) and the hidden states are computed, the output arrays are added to
        # the prediction array and the hidden states are added to the hidden states array for later use in the truncated
        # back propagation through time.
        for i in range (len(Y)):
            u, y = U[i], Y[i]
            x = self.state(x, u)
            yHat = np.dot(self.Woutput, x)
            yHat = self.sigmoid(yHat)
            hidden_states.append(x)
            preds.append(yHat)
        return np.array(preds), np.array(hidden_states)

    # A method that performs truncated back propagation through time. Truncated is used to make it less computationally
    # expensive. The gradients of the weight matrices are computed with the use of the derivative of the loss.
    def backprop(self, yHat, U, x, Y):
        # initialization of the gradients with zeros:
        dWin = np.zeros(self.Winput.shape)
        dWout = np.zeros(self.Woutput.shape)
        dW = np.zeros(self.W.shape)

        # computation of the derivative of the loss:
        delta_loss = 2 * (abs(yHat - Y))
        delta_loss = np.array(delta_loss)

        # for each computed output, the gradients are computed.
        for t in range(len(Y))[::-1]:
            dWout += np.outer(delta_loss[t], x[t])
            delta_t = np.dot(np.transpose(self.Woutput), delta_loss[t])
            delta_t *= self.sigmoidPrime(x[t])

            # the truncated loop, it does not go back through all the states but it goes back as many steps as the
            # bptt_truncate value:
            maximum = max(0, t-self.bptt_truncate)
            for timestep in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dW += np.outer(delta_t, x[timestep - 1])
                dWin += np.outer(delta_t, U[timestep])
                delta_t = np.dot(np.transpose(self.W), delta_t)
                delta_t *= self.sigmoidPrime(x[timestep-1])

            # averaging of the gradients:
            dW /= (t+1-maximum)
            dWin /= (t+1-maximum)

        dWin /= len(Y)
        dW /= len(Y)
        dWout /= len(Y)
        return dWin, dW, dWout

    # A method that ?????????????????????????????
    def lastWUpdate(self, Y):
        self.Winput += (2 * self.alfa * (self.Winput / len(Y)))
        self.W +=  (2 * self.alfa * (self.W / len(Y)))
        self.Woutput += (2 * self.alfa * (self.Woutput / len(Y)))
        print("winput", self.Winput, "W", self.W, "Woutput", self.Woutput)

    # A method that updates the weight matrices.
    def updateWeights(self, dWin, dW, dWout, Y):
        #preventing from exploding gradient problem:
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

        #ridge regression????????????????????????????????????????????????????????????????????????:
        (2 * self.alfa * (self.Winput / len(Y)))
        self.W +=  (2 * self.alfa * (self.W / len(Y)))
        self.Woutput += (2 * self.alfa * (self.Woutput / len(Y)))
        #updating:
        print("winput", self.Winput, "W", self.W, "Woutput", self.Woutput)
        self.Winput = self.Winput - self.learning_rate * dWin + 2 * self.alfa * (self.Winput / len(Y))
        self.W = self.W - self.learning_rate * dW + self.W + 2 * self.alfa * (self.W / len(Y))
        self.Woutput = self.Woutput - self.learning_rate * dWout + 2 * self.alfa * (self.Woutput / len(Y))

    # A method that trains the recurrent neural network by performing the forward pass and the truncated backpropagation
    # through time and that updates the weight matrices.
    def training(self, U, Y):
        x = np.zeros((self.hidden_size, 1))
        yHat, hidden_states = self.forward(U, x, Y)
        yHat = np.reshape(yHat, (len(Y), self.vocab_size))
        hidden_states = np.reshape(hidden_states, (len(Y), self.hidden_size))
        dWin, dW, dWout = self.backprop(yHat, U, hidden_states, Y)
        self.updateWeights(dWin, dW, dWout, Y)

    # A method that computes an output with a prime array.
    def prediction(self, U):
        hidden_states = []
        preds = []
        x = np.zeros((self.hidden_size, 1))
        # computation of the outputs with the use of the prime array, this goes the same way as in the forward function.
        for i in range (len(U)):
            u = U[i]
            print("x", x)
            x = self.state(x, u)

            yHat = np.dot(self.Woutput, x)
            yHat = self.sigmoid(yHat)
            hidden_states.append(x)
            print("YHat: ",yHat)
            preds.append(yHat)
        # computation of the outputs in which the previous output is used as the new input.
        for j in range(len(U), 100):
            x = self.state(x, yHat)
            yHat = np.dot(self.Woutput, x)
            yHat = self.sigmoid(yHat)
            hidden_states.append(x)
            preds.append(yHat)
        return np.array(preds)


# a recurrent neural network is initialized and copied to be used for cross validation.
rnn = RNN()
rnnRun = rnn
print("trainu :", vector_array_u_train)
print("testu :", vector_array_u_test)

# a loop to compute the number of epochs that prevents from overfitting. This number of epochs is computed by increasing
# the number until the testing loss no longer decreases.
epoch = 0
previous_Testloss = rnn.checkLoss(vector_array_u_test, vector_array_y_test)
testLoss = previous_Testloss
trainLoss = rnn.checkLoss(vector_array_u_train, vector_array_y_train)
prev_loss = trainLoss
testLosses = []
trainingLosses = []
while (previous_Testloss[0] + previous_Testloss[1] >= testLoss[0] + testLoss[1]):
    epoch += 1
    trainingLosses.append(trainLoss[0] + trainLoss[1])
    testLosses.append(testLoss[0] + testLoss[1])
    prev_loss = trainLoss
    previous_Testloss = testLoss
    rnn.training(vector_array_u_train, vector_array_y_train)
    trainLoss = rnn.checkLoss(vector_array_u_train, vector_array_y_train)
    testLoss = rnn.checkLoss(vector_array_u_test, vector_array_y_test)
    print('Epoch: ', epoch , ', Loss: ', trainLoss, ', Val Loss: ', testLoss)

#rnn.lastWUpdate(vector_array_y_train)
#plot_losses(testLosses, trainingLosses, epoch)

vector_array_u_test_train = np.append(vector_array_u_test, vector_array_u_train, axis=0)
vector_array_y_test_train = np.append(vector_array_y_train, vector_array_y_test, axis=0)
print("trainu :", vector_array_u_train)
print("testtrainu :", vector_array_u_test_train)

# the actual training of the recurrent neural network with the right number of epochs.
for i in range(epoch):
    rnnRun.training(vector_array_u_test_train, vector_array_y_test_train)
    loss = rnn.checkLoss(vector_array_u_test_train, vector_array_y_test_train)
    print('Epoch: ', i, ', Loss: ', loss)

#RnnRun.lastWUpdate(vector_array_y_test_train)
    #rnnRun.Winput += rnnRun.alfa * rnnRun.Winput
#rnnRun.W += rnnRun.alfa * np.dot(rnnRun.W, np.transpose(rnnRun.W))
#rnnRun.Woutput += rnnRun.alfa * rnnRun.Woutput

# the prime array is created with a part from the testing and training array:
vector_array_prime = []
for i in range(10, 30):
    vector_array_prime.append(vector_array_u_test_train[i])
vector_array_prime = np.array(vector_array_prime)

# the computation of the output with the use of the prime array:
result = rnnRun.prediction(vector_array_prime)
result = np.reshape(result, (len(result), rnnRun.vocab_size))

for i in range(0, len(result)):
    y = result[i]
    loudness1 = int(y[0] * 127)
    loudness2 = int(y[1] * 127)
    print(loudness1, loudness2)

print("result :",result)
print("shape result: ",result.shape)

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

