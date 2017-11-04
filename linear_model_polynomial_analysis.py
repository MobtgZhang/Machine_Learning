from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,Lars,OrthogonalMatchingPursuit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# make random data
rng = np.random.RandomState(384)
x = np.linspace(-10,10,200)[:,np.newaxis]
tip = 200
noise = tip * rng.uniform(0,1,x.shape)

real_data = np.power(x,3) - 100* x + 600
y =  real_data + noise
# draw points
plt.scatter(x,y,c = "r")
# using Lasso algorithm
myclass = Lasso()

model = LinearRegression(fit_intercept = True,normalize = False,copy_X = True,n_jobs = 3)
model.fit(x,y)
poly_model = make_pipeline(PolynomialFeatures(degree = 3,interaction_only = False,include_bias = True)
							,myclass)
poly_model.fit(x,y)
zB = poly_model.predict(x)
zA = model.predict(x)

score1 = model.score(x,real_data)
score2 = poly_model.score(x,real_data)

plt.plot(x,zA,c = "y",lw = 3,label = "Linear Regression")
plt.plot(x,zB,c = "b",lw = 3,label = "PolynomialFeatures Regression")
plt.legend(loc = "best")
plt.title("The LinearRegression score:%f\nThe PolynomialFeatures score:%f"%(score1,score2))
plt.show()