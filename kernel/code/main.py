from DataReader import prepare_data
from model import Model

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def main():
    # ------------Data Preprocessing------------
    train_X, train_y, valid_X, valid_y, train_valid_X, train_valid_y, test_X, test_y = prepare_data(data_dir, train_filename, test_filename)

    # ------------Kernel Logistic Regression Case------------
    ### YOUR CODE HERE
    # Run your kernel logistic regression model here
    learning_rate = [0.01, 0.001]
    max_epoch = [50]
    batch_size = [8, 16, 32]
    sigma = [1, 2, 3]
    scores = []
    
    for lr in learning_rate:
        for ep in max_epoch:
            for bs in batch_size:
                for sig in sigma:
                    model = Model('Kernel_LR', train_X.shape[0], sig)
                    model.train(train_X, train_y, valid_X, valid_y, ep, lr, bs)

                    model = Model('Kernel_LR', train_valid_X.shape[0], sig)
                    model.train(train_valid_X, train_valid_y, None, None, ep, lr, bs)
                    score = model.score(test_X, test_y)
                    res = "lr = " + str(lr) + ", epochs = " + str(ep) + ", batch_size = " + str(bs) + ", sigma = " + str(sig) + ", test score = " + str(score)
                    scores.append(res)

    for s in scores:
        print(s)

    ### END YOUR CODE
    
    # ------------RBF Network Case------------
    ### YOUR CODE HERE
    learning_rate = [0.01, 0.001]
    max_epoch = [50]
    batch_size = [32, 64]
    sigma = [0.1, 1, 3]
    scores = []
    hidden_dims = [8, 16, 32]

    for lr in learning_rate:
        for ep in max_epoch:
            for bs in batch_size:
                for sig in sigma:
                    for hd in hidden_dims:
                        model = Model('RBF', hd, sig)
                        model.train(train_X, train_y, valid_X, valid_y, ep, lr, bs)
                        model = Model('RBF', hd, sig)
                        model.train(train_valid_X, train_valid_y, None, None, ep, lr, bs)
                        score = model.score(test_X, test_y)
                        res = "lr = " + str(lr) + ", epochs = " + str(ep) + ", batch_size = " + str(bs) + ", sigma = " + str(sig) + ", hidden dimension = " + str(hd) + ", test score = " + str(score)
                        scores.append(res)
    for s in scores:
        print(s)
    ### END YOUR CODE
    
    # ------------Feed-Forward Network Case------------
    ### YOUR CODE HERE
    learning_rate = [0.01, 0.001]
    max_epoch = [50]
    batch_size = [32, 64]
    scores = []
    hidden_dims = [8, 16, 32]

    for lr in learning_rate:
        for ep in max_epoch:
            for bs in batch_size:
                for hd in hidden_dims:
                    model = Model('FFN', hd)
                    model.train(train_X, train_y, valid_X, valid_y, ep, lr, bs)
                    model = Model('FFN', hd)
                    model.train(train_valid_X, train_valid_y, None, None, ep, lr, bs)
                    score = model.score(test_X, test_y)
                    res = "lr = " + str(lr) + ", epochs = " + str(ep) + ", batch_size = " + str(bs) + ", hidden dimension = " + str(hd) + ", test score = " + str(score)
                    scores.append(res)
    for s in scores:
        print(s)
    ### END YOUR CODE

if __name__ == '__main__':
    main()