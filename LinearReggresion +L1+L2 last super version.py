import numpy as np
import matplotlib.pyplot as plt
class LinearRegression:
    def __init__(self,lr=0.01,n_iters=1000, reg = None, alfa=0.1):
        """
        lr - скорость обуч
        n_iters - колич итераций
        reg - регуляризация Л1 или Л2
        alfa - коэф регуляризации
        """
        self.lr = lr
        self.n_iters=n_iters
        self.reg = reg
        self.alfa = alfa
        self.weights = None
        self.Bias = None
    def fit(self,X,y):
        X = np.asarray(X,dtype=float)
        y = np.asarray(y,dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1,1)
        n_samples,n_features = X.shape
        self.weights = np.zeros_like(X[0],dtype=float)
        self.Bias=0.0
        for i in range(self.n_iters):
            y_pred = X @ self.weights + self.Bias
            error = y_pred - y
            dw = (X.T @ error)/ n_samples
            db = np.mean(error)
            if self.reg == 'l2' or self.reg == 'L2':
                dw+= (self.alfa/ n_samples)*self.weights
            elif self.reg == 'l1' or self.reg == 'L1':
                dw+= (self.alfa/n_samples) * np.sign(self.weights)
            self.weights -= self.lr* dw
            self.Bias -= self.lr * db
        return self
    def predict(self,X):
        X = np.asarray(X,dtype=float)
        if X.ndim ==1:
            X = X.reshape(-1,1)
        return X @ self.weights + self.Bias
    def mse (self,X,y):
        X = np.asarray(X,dtype=float)
        y = np.asarray(y,dtype=float).ravel()
        y_pred = self.predict(X)
        return np.mean((y_pred-y)**2)
    def r2(self,X,y):
        X = np.asarray(X,dtype=float)
        y = np.asarray(y,dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res= np.sum((y - y_pred)**2)
        ss_tot = np.sum((y-y.mean())**2)
        return 1 - ss_res/ss_tot if ss_tot !=0 else 0.0
rng = np.random.default_rng(100)
X = 2* rng.random((100,1))
y = 4+ 3*X[:,0]+ rng.normal(0,1,size=100)
lr_clear= LinearRegression(lr=0.1,n_iters = 2000,reg = None)
lr_clear.fit(X,y)
lr_ridge = LinearRegression(lr =0.1,n_iters = 3000,reg = 'l2', alfa = 0.5)
lr_ridge.fit(X,y)
lr_lasso = LinearRegression(lr = 0.1,n_iters = 4000,reg = 'l1',alfa = 0.5)
lr_lasso.fit(X,y)
plt.figure(figsize=(10,6))
plt.scatter(X,y,color = 'gray',alpha = 0.6,label ='данные')
X_line = np.linspace(X.min(),X.max(),100).reshape(-1,1)
plt.plot(X_line,lr_clear.predict(X_line),color= 'red',label = 'без регул')
plt.plot(X_line,lr_ridge.predict(X_line),color= 'blue',label = 'л2')
plt.plot(X_line,lr_lasso.predict(X_line),color= 'green',label = 'л1')
plt.xlabel('X')
plt.ylabel('y')
plt.title("LinReg + L1 + L2")
plt.legend()
plt.grid(True)
plt.show()










