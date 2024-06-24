import numpy as np

class LinearRegression:

    def __init__(self, in_feat, out_feat):
        self.W = np.ones((in_feat, out_feat))
        self.B = np.ones(out_feat)

    def forward(self, X):
        return X@self.W + self.B

    def fit(self, X, Y, lr = 5e-5, epochs = 200):
        #using gradient descent
        #and MNE

        n = X.shape[0] #Number of data
        
        for e in range(epochs):
            Y_pred = self.forward(X)
            print(f"loss: {MNE(Y,Y_pred)}")
            # print(Y_pred, Y_pred.shape)

            dY = Y - Y_pred
            # print(dY.T@X/n)
            # print(self.W.shape)
            W_grad = -(dY.T@X).T /n
            B_grad = -np.sum(dY, axis = 0) /n

            self.W -= lr*W_grad
            self.B -= lr*B_grad



def MNE(Y, Y_pred):
    return sum(map(lambda x: x**2,Y-Y_pred))/ Y.shape[0]

