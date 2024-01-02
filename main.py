import torch as t
from torchvision import datasets
import pandas as pd
from exceptions import tooSmall
from model import Model
import sys
import numpy as np

try:

    DATA_SET_SIZE = 28*28
    BATCH_PERCENT = 0.45
    EPOCHS = 500
    LEARNING_RATE = 0.0001
    FEATURES_NUMBER = 1
    NOISE = 0.02
    ACCURACY_COEFF = 0.3

    DATA_SET_SIZE = round(DATA_SET_SIZE)
    EPOCHS = round(EPOCHS)
    BATCH_SIZE = round(DATA_SET_SIZE * BATCH_PERCENT)
    FEATURES_NUMBER = round(FEATURES_NUMBER)

    exceptionz = []
    exception1 = DATA_SET_SIZE >= 1
    exceptionz.append(exception1)
    exception2 = 0 < BATCH_PERCENT <= 1
    exceptionz.append(exception2)
    exception3 = EPOCHS >= 1
    exceptionz.append(exception3)
    exception4 = BATCH_SIZE >= 1
    exceptionz.append(exception4)
    exception5 = 0 < LEARNING_RATE <= 1
    exceptionz.append(exception5)
    exception6 = FEATURES_NUMBER >= 1
    exceptionz.append(exception6)
    exception7 = 0 <= NOISE <= 1
    exceptionz.append(exception7)
    exception8 = 0 < ACCURACY_COEFF < 1
    exceptionz.append(exception8)
    exceptionz = np.array(exceptionz)

    if not exceptionz.any():
        raise tooSmall(name='ALL')
    if (exceptionz == False).sum() >= 2:
        raise tooSmall(name='ANY')
    if not exception1:
        raise tooSmall(name='DATA_SET_SIZE', b=DATA_SET_SIZE)
    if not exception2:
        raise tooSmall(name='BATCH_PERCENT', b=BATCH_PERCENT)
    if not exception3:
        raise tooSmall(name='EPOCHS', b=EPOCHS)
    if not exception4:
        raise tooSmall(name='BATCH_SIZE', b=BATCH_SIZE)
    if not exception5:
        raise tooSmall(name='LEARNING_RATE', b=LEARNING_RATE)
    if not exception6:
        raise tooSmall(name='FEATURES_NUMBER', b=FEATURES_NUMBER)
    if not exception7:
        raise tooSmall(name='NOISE', b=NOISE)
    if not exception8:
        raise tooSmall(name='ACCURACY_COEFF', a=0.9999999999, b=ACCURACY_COEFF)

except tooSmall as e:
    if e.name == 'ALL':
        DATA_SET_SIZE = e.b
        BATCH_PERCENT = e.b
        EPOCHS = e.b
        BATCH_SIZE = e.b
        LEARNING_RATE = e.b
        NOISE = e.b
        ACCURACY_COEFF = e.b - 0.0000000001
    if e.name == 'DATA_SET_SIZE':
        DATA_SET_SIZE = e.b
    if e.name == 'BATCH_PERCENT':
        BATCH_PERCENT = e.b
    if e.name == 'EPOCHS':
        EPOCHS = e.b
    if e.name == 'BATCH_SIZE':
        BATCH_SIZE = e.b
    if e.name == 'LEARNING_RATE':
        LEARNING_RATE = e.b
    if e.name == 'NOISE':
        NOISE = e.b
    if e.name == 'FEATURES_NUMBER':
        FEATURES_NUMBER = e.b
    if e.name == 'ACCURACY_COEFF':
        ACCURACY_COEFF = e.b - 0.0000000001
    if e.name == 'ANY':
        sys.exit(1)
finally:
    # Loading train data:
    # train = datasets.MNIST(root="./data", train=True)
    train = pd.read_csv("train.csv")

    # Creating the model:
    mod = Model(DATA_SET_SIZE, 10)

    # ...Optimizer:
    optimizer = t.optim.Adam(params=mod.parameters(), lr=LEARNING_RATE)

    # ...and Criterion:
    criterion = t.nn.CrossEntropyLoss()

    acc = []

    for epoch in range(train.shape[0]):
        # Load 1 image:
        # data, label = train[epoch]
        data = train.iloc[epoch, 1:]
        label = train.iloc[epoch, 0]

        # Turning it into a np array:
        data = np.array(data)
        label = np.array([label])

        # Then into a tensor:
        data = t.from_numpy(data.astype(np.float32))
        label = t.from_numpy(label.astype(np.float32))
        label = label.long()

        # Then we resize it:
        data = data.view(-1, DATA_SET_SIZE)

        # Finally, we require gradient:
        data.requires_grad_(True)

        # Setting the optimizer:
        optimizer.zero_grad()

        # Inserting input to the model:
        result = mod(data)

        # Loss:
        loss = criterion(result, label)

        # Fitting:
        loss.backward()
        optimizer.step()

        # Accuracy appending:
        acc.append(result.argmax() == label)

        if epoch % (0.1*train.shape[0]) == 0:
            print(np.array(acc).sum() / len(acc))
            acc = []

    # We set the model to evaluation:
    mod.eval()

    # We define an accuracy counter:
    # acc = []

    # And the results array:
    results = []

    # Loading train data:
    # test = datasets.MNIST(root="./data", train=False, download=True)
    test = pd.read_csv("test.csv")

    print("test")

    for epoch in range(test.shape[0]):
        # Load 1 image:
        # data, label = test[epoch]
        data = test.iloc[epoch, :]

        # Turning it into a np array:
        data = np.array(data)

        # Then into a tensor:
        data = t.from_numpy(data.astype(np.float32))

        # Then we resize it:
        data = data.view(-1, DATA_SET_SIZE)

        # Inserting input to the model:
        with t.no_grad():
            result = mod(data)
            results.append(result.argmax().item())

        # Accuracy appending:
        # acc.append(result.argmax() == label)

        # if epoch % (0.1*train.shape[0]) == 0:
        #     print(np.array(acc).sum() / len(acc))
        #     acc = []

    # Finally, we save the results as a csv:
    output = pd.DataFrame()
    output['ImageId'] = range(1, test.shape[0] + 1)
    output['Label'] = results
    output.to_csv('output.csv', index=False)
