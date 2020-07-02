import numpy as np
import csv

U = np.empty(shape=(1006, 2, 1))
Y = np.empty(shape=(1006, 2, 1))
vector = np.empty(shape=(2, 1, 1))

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

        vector = np.array([[int(loudness1)/127, int(loudness2)/127]])
        U[i] = vector.T

        loudness1 = 0
        loudness2 = 0

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

        vector = np.array([[int(loudness1)/127, int(loudness2)/127]])
        U[i] = vector.T

        loudness1 = 0
        loudness2 = 0

    for j in range(1, len(U)):
        Y[j-1] = U[j]

U_val = np.empty(shape=(1006, 2, 1))
Y_val = np.empty(shape=(1006, 2, 1))

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

        vector = np.array([[int(loudness1)/127, int(loudness2)/127]])
        U_val[i] = vector.T

        loudness1 = 0
        loudness2 = 0

    for j in range(1, len(U_val)):
        Y_val[j - 1] = U_val[j]

learning_rate = 0.0001
nepoch = 50
T = 1006
hidden_dim = 170
output_dim = 2
input_dim = 2

bptt_truncate = 5
min_clip_value = -10
max_clip_value = 10

Win = np.random.uniform(0, 1, (hidden_dim, input_dim))
Wh = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
Wout = np.random.uniform(0, 1, (output_dim, hidden_dim))

b = np.ones((hidden_dim, 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

'''Step 2.1 : Check the loss on training data
Step 2.1.1 : Forward Pass
Step 2.1.2 : Calculate Error
Step 2.2 : Check the loss on validation data
Step 2.2.1 : Forward Pass
Step 2.2.2 : Calculate Error
Step 2.3 : Start actual training
Step 2.3.1 : Forward Pass
Step 2.3.2 : Backpropagate Error
Step 2.3.3 : Update weights'''

#step 2.1:

for epoch in range(nepoch):
    # check loss on train
    loss = 0.0
    layers = []

    # do a forward pass to get prediction
    for i in range(len(Y)):
        u, y = U[i], Y[i]  # get input, output values of each record
        prev_x = np.zeros((hidden_dim, 1))  # here, prev-s is the value of the previous activation of hidden layer; which is initialized as all zeroes

        x = np.zeros((hidden_dim, 1))  # we then do a forward pass for every timestep in the sequence


        prev_x = np.zeros((hidden_dim, 1))

        dWin = np.zeros(Win.shape)
        dWout = np.zeros(Wout.shape)
        dWh = np.zeros(Wh.shape)

        dWin_t = np.zeros(Win.shape)
        dWout_t = np.zeros(Wout.shape)
        dWh_t = np.zeros(Wh.shape)

        dWin_i = np.zeros(Win.shape)
        dWh_i = np.zeros(Wh.shape)

        mulWin = np.dot(Win, u)
        mulWh = np.dot(Wh, prev_x)
        add = mulWh + mulWin + b
        x = sigmoid(add)
        yhat = np.dot(Wout, x)
        layers.append({'x': x, 'prev_x': prev_x})

        prev_x = x

        # calculate error
        loss_per_record = np.dot((y - yhat).T, (y - yhat)) / 2
        loss += loss_per_record

    loss = loss / float(y.shape[0])

    # derivative of pred
    dmulWout = (yhat - y)


    # backward pass
    for t in range(len(Y)):
        dWout_t = np.dot(dmulWout, np.transpose(layers[t]['x']))
        dxWout = np.dot(np.transpose(Wout), dmulWout)

        dx = dxWout
        dadd = add * (1 - add) * dx

        dmulWh = dadd * np.ones_like(mulWh)

        dprev_x = np.dot(np.transpose(Wh), dmulWh)

        for i in range(t - 1, max(-1, t - bptt_truncate - 1), -1):
            dx = dxWout + dprev_x
            dadd = add * (1 - add) * dx

            dmulWh = dadd * np.ones_like(mulWh)
            dmulWin = dadd * np.ones_like(mulWin)

            dWh_i = np.dot(Wh, layers[t]['prev_x'])
            dprev_x = np.dot(np.transpose(Wh), dmulWh)

            dWin_i = np.dot(Win, U[i])
            dx = np.dot(np.transpose(Win), dmulWin)

            dWin_t += dWin_i
            dWh_t += dWh_i

        dWout += dWout_t
        dWin += dWin_t
        dWh += dWh_t

    if dWin.max() > max_clip_value:
        dWin[dWin > max_clip_value] = max_clip_value
    if dWout.max() > max_clip_value:
        dWout[dWout > max_clip_value] = max_clip_value
    if dWh.max() > max_clip_value:
        dWh[dWh > max_clip_value] = max_clip_value

    if dWin.min() < min_clip_value:
        dWin[dWin < min_clip_value] = min_clip_value
    if dWout.min() < min_clip_value:
        dWout[dWout < min_clip_value] = min_clip_value
    if dWh.min() < min_clip_value:
        dWh[dWh < min_clip_value] = min_clip_value

        # update
    Win -= learning_rate * dWin
    Wout -= learning_rate * dWout
    Wh -= learning_rate * dWh

    preds = []
    for i in range(len(Y)):
        u, y = U[i], Y[i]
        prev_x = np.zeros((hidden_dim, 1))
        # Forward pass
        mulu = np.dot(Win, u)
        mulw = np.dot(Wh, prev_x)
        add = mulw + mulu + b
        s = sigmoid(add)
        yhat = np.dot(Wout, x)
        prev_s = s

        preds.append(yhat)

    preds = np.array(preds)
    print(preds)

    # check loss on val
    val_loss = 0.0
    for i in range(Y_val.shape[0]):
        u, y = U_val[i], Y_val[i]
        prev_x = np.zeros((hidden_dim, 1))

        x = np.zeros((2, 1))

        mulWin = np.dot(Win, u)
        mulW = np.dot(Wh, prev_x)
        add = mulW + mulWin + b
        x = sigmoid(add)
        yhat = np.dot(Wout, x)
        prev_x = x

        loss_per_record = np.dot((y - yhat).T, (y-yhat)) / 2
        val_loss += loss_per_record
    val_loss = val_loss / float(y.shape[0])

    print('Epoch: ', epoch + 1, ', Loss: ', loss, ', Val Loss: ', val_loss)