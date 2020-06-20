import numpy as np
import math
import re
import csv
#import matplotlib.pyplot as plt
import py_midicsv

vector_array_u_train = []
vector_array_y_train = []
with open('sunday.csv', newline='') as f: # andere csv file openen
    reader = csv.reader(f)
    data = [r for r in reader]

    line_count = 0
    loudness1 = 0
    loudness2 = 0
    quarter_notes = int(data[line_count][1]) / 192 # verandert per liedje
    sixteenth_notes = quarter_notes
    for i in range(0, 464): # range end verandert ook
        while int(sixteenth_notes) == i:

            if int(data[line_count][4]) == 38:
                loudness1 = data[line_count][5]
            if int(data[line_count][4]) == 42:
                loudness2 = data[line_count][5]
            line_count += 1

            quarter_notes = int(data[line_count][1]) / 192 # verandert per liedje
            sixteenth_notes = quarter_notes

        vector = [float(loudness1)/127, float(loudness2)/127]
        print(vector)
        vector_array_u_train.append(vector)
        loudness1 = 0
        loudness2 = 0


with open('follow.csv', newline='') as f: # andere csv file openen
    reader = csv.reader(f)
    data = [r for r in reader]

    line_count = 0
    loudness1 = 0
    loudness2 = 0
    quarter_notes = int(data[line_count][1]) / 120 # verandert per liedje
    sixteenth_notes = quarter_notes
    for i in range(0, 543): # range end verandert ook
        while int(sixteenth_notes) == i:

            if int(data[line_count][4]) == 38:
                loudness1 = data[line_count][5]
            if int(data[line_count][4]) == 42:
                loudness2 = data[line_count][5]
            line_count += 1

            quarter_notes = int(data[line_count][1]) / 120 # verandert per liedje
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

with open('pride.csv', newline='') as f: # andere csv file openen
    reader = csv.reader(f)
    data = [r for r in reader]

    line_count = 0
    loudness1 = 0
    loudness2 = 0
    quarter_notes = int(data[line_count][1]) / 480 # verandert per liedje
    sixteenth_notes = quarter_notes
    for i in range(0, 407): # range end verandert ook
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

def plot_losses(testingLoss, trainingLoss, epoch):
    t = np.linspace(0.0, epoch, epoch)
    fig, ax = plt.subplots()
    ax.plot(t, testingLoss, label = "loss")
    ax.plot(t, trainingLoss, label = "Emperical loss")

    ax.set(xlabel='Epochs', ylabel='Loss/emperical risk', title='Model flexibility', label='')
    ax.grid()

    fig.savefig("flexibility.png")
    plt.show()




class RNN:
    def __init__(self):

        self.hidden_size = 500
        self.vocab_size = 2
        self.learning_rate = 0.001

        self.bptt_truncate = 5
        self.min_clip_value = -1   #uitproberen met trial and error
        self.max_clip_value = 1    #uitproberen met trial and error
        self.alfa = 2

        self.Winput = np.random.uniform(-np.sqrt(1./self.vocab_size), np.sqrt(1./self.vocab_size), (self.hidden_size, self.vocab_size))
        self.W = np.random.uniform(-np.sqrt(1./self.vocab_size), np.sqrt(1./self.vocab_size), (self.hidden_size, self.hidden_size))
        self.Woutput = np.random.uniform(-np.sqrt(1./self.vocab_size), np.sqrt(1./self.vocab_size), (self.vocab_size, self.hidden_size))

        self.b = np.ones((self.hidden_size, 1))

    def sigmoidPrime(self, x):
        y = self.sigmoid(x)
        return y*(1-y)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def state(self, x, u):
        u = np.reshape(u, (2,1))
        mulu = np.dot(self.Winput, u)
        mulx = np.dot(self.W, x)
        add = np.add(mulu, mulx)
        lol = self.sigmoid(np.add(add, self.b))
        return np.array(lol)

    def checkLoss(self, U, Y):
        x = np.zeros((self.hidden_size, 1))
        YHat, _ = self.forward(U, x, Y)
        loss = np.zeros((self.vocab_size, 1))
        for i in range(len(Y)):
            yHat, y = YHat[i], Y[i]
            yHat = np.reshape(yHat, (2, 1))
            y = np.reshape(y, (2, 1))

            loss += (abs(y-yHat))**2
        risk = loss / len(Y)
        return risk

    def forward(self, U, x, Y):
        preds = []
        hidden_states = []
        for i in range (len(Y)): #lenY is the number of samples
            u, y = U[i], Y[i]
            x = self.state(x, u)
            #print("xshape", x.shape)
            yHat = np.dot(self.Woutput, x)
            yHat = self.sigmoid(yHat)
            hidden_states.append(x)
            preds.append(yHat)
        return np.array(preds), np.array(hidden_states)

    def backprop(self, yHat, U, x, Y):
        dWin = np.zeros(self.Winput.shape)
        dWout = np.zeros(self.Woutput.shape)
        dW = np.zeros(self.W.shape)

        delta_loss = 2 * (abs(yHat - Y))
        delta_loss = np.array(delta_loss)

        for t in range(len(Y))[::-1]: #deze loop gaat van len(Y) met stapjes van 1 naar 0
            dWout += np.outer(delta_loss[t], x[t])  #outer klopt en transpose niet nodig
            delta_t = np.dot(np.transpose(self.Woutput), delta_loss[t])
            delta_t *= self.sigmoidPrime(x[t]) # we doen hier toch * want dot ging fout (is een getal niet een vector)

            maximum = max(0, t-self.bptt_truncate)
            for timestep in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dW += np.outer(delta_t, x[timestep - 1])
                dWin += np.outer(delta_t, U[timestep]) # ik ben hier niet helemaal zeker van maar lijkt wel logisch
                delta_t = np.dot(np.transpose(self.W), delta_t)
                delta_t *= self.sigmoidPrime(x[timestep-1])

            dW /= (t+1-maximum)
            dWin /= (t+1-maximum)

        dWin /= len(Y)
        dW /= len(Y)
        dWout /= len(Y)
        return dWin, dW, dWout

    def lastWUpdate(self, Y):
        self.Winput += (2 * self.alfa * (self.Winput / len(Y)))
        self.W +=  (2 * self.alfa * (self.W / len(Y)))
        self.Woutput += (2 * self.alfa * (self.Woutput / len(Y)))
        print("winput", self.Winput, "W", self.W, "Woutput", self.Woutput)

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

        (2 * self.alfa * (self.Winput / len(Y)))
        self.W +=  (2 * self.alfa * (self.W / len(Y)))
        self.Woutput += (2 * self.alfa * (self.Woutput / len(Y)))
        #updating: + alfa w'w???
        print("winput", self.Winput, "W", self.W, "Woutput", self.Woutput)
        self.Winput = self.Winput - self.learning_rate * dWin + 2 * self.alfa * (self.Winput / len(Y))
        self.W = self.W - self.learning_rate * dW + self.W + 2 * self.alfa * (self.W / len(Y))
        self.Woutput = self.Woutput - self.learning_rate * dWout + 2 * self.alfa * (self.Woutput / len(Y))

    def training(self, U, Y):
        x = np.zeros((self.hidden_size, 1))
        yHat, hidden_states = self.forward(U, x, Y)
        yHat = np.reshape(yHat, (len(Y), self.vocab_size))
        hidden_states = np.reshape(hidden_states, (len(Y), self.hidden_size))
        dWin, dW, dWout = self.backprop(yHat, U, hidden_states, Y)
        self.updateWeights(dWin, dW, dWout, Y)

    def prediction(self, U): #moeten we niet primen????
        #print("W: ",self.W)
        #print("Win: ", self.Winput)
        #print("Wout: ", self.Woutput)
        hidden_states = []
        preds = []
        x = np.zeros((self.hidden_size, 1))
        for i in range (len(U)):
            u = U[i]
            print("x", x)
            x = self.state(x, u)

            yHat = np.dot(self.Woutput, x)
            yHat = self.sigmoid(yHat)
            hidden_states.append(x)
            print("YHat: ",yHat)
            preds.append(yHat)
        for j in range(len(U), 100):
            x = self.state(x, yHat)
            yHat = np.dot(self.Woutput, x)
            yHat = self.sigmoid(yHat)
            hidden_states.append(x)
            preds.append(yHat)
        return np.array(preds)


rnn = RNN()
rnnRun = rnn
print("trainu :", vector_array_u_train)
print("testu :", vector_array_u_test)

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

for i in range(epoch):
    rnnRun.training(vector_array_u_test_train, vector_array_y_test_train)
    loss = rnn.checkLoss(vector_array_u_test_train, vector_array_y_test_train)
    print('Epoch: ', i, ', Loss: ', loss)

#RnnRun.lastWUpdate(vector_array_y_test_train)
    #rnnRun.Winput += rnnRun.alfa * rnnRun.Winput
#rnnRun.W += rnnRun.alfa * np.dot(rnnRun.W, np.transpose(rnnRun.W))
#rnnRun.Woutput += rnnRun.alfa * rnnRun.Woutput

vector_array_prime = []

for i in range(10, 30):
    vector_array_prime.append(vector_array_u_test_train[i])

vector_array_prime = np.array(vector_array_prime)

result = rnnRun.prediction(vector_array_prime)
result = np.reshape(result, (len(result), rnnRun.vocab_size))

for i in range(0, len(result)):
    y = result[i]
    loudness1 = int(y[0] * 127)
    loudness2 = int(y[1] * 127)
    print(loudness1, loudness2)

#print("result :",result)
print("shape result: ",result.shape)

#with open('result.csv', 'w') as f:
#    f.write('0, 0, Header, 1, 2, 264\n')
#    f.write('1, 0, Start_track\n')
#    f.write('1, 0, Key_signature, 0, "major"\n')
#    f.write('1, 0, Tempo, 501133\n')
#    f.write('1, 0, Time_signature, 4, 2, 24, 8\n')
#    f.write('1, 0, End_track\n')
#    f.write('2, 0, Start_track\n')
#    f.write('2, 0, Title_t, "drums"\n')
#    f.write('2, 0, MIDI_port, 0\n')
#    for i in range(0, len(result)):
#        y = result[i]
#        midi_clock = i * 264
#
#        loudness1 = int(y[0] * 127)
#        loudness2 = int(y[1] * 127)
#        f.write('2, {}, Note_on_c, 9, 38, {}\n'.format(midi_clock, loudness1))
#        f.write('2, {}, Note_on_c, 9, 42, {}\n'.format(midi_clock, loudness2))
#    f.write('2, 0, End_track\n')
#    f.write('0, 0, End_of_file\n')


#midi = py_midicsv.csv_to_midi('result.csv')

#with open("result.mid", "wb") as output_file:
#    midi_writer = py_midicsv.FileWriter(output_file)
#    midi_writer.write(midi)
#
