import theano
import theano.tensor as T
import numpy as np
class BPClassification:
    def __init__(self,in_dim,hid_dim,out_dim):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        W_i = np.random.rand(in_dim,hid_dim)
        b_i = np.random.rand(hid_dim)
        W_o = np.random.rand(hid_dim,out_dim)
        b_o = np.random.rand(out_dim)
        self.W_i = theano.shared(value=W_i,name="W_i")
        self.b_i = theano.shared(value=b_i,name="b_i")
        self.W_o = theano.shared(value=W_o,name="W_o")
        self.b_o = theano.shared(value=b_o,name="b_o")

        self.build_model()
    def build_model(self):
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        W_i = self.W_i
        b_i = self.b_i
        W_o = self.W_o
        b_o = self.b_o
        # forward
        hid = T.nnet.relu(T.dot(input,W_i)+b_i,0)
        z_out = T.dot(hid,W_o) + b_o
        output = T.nnet.softmax(z_out)
        self.forward = theano.function(inputs=[input],
                                       outputs=output)
        o_error = T.sum(T.nnet.categorical_crossentropy(output,target))
        length = output.shape[0]
        o_error = o_error/length
        self.loss = theano.function(inputs=[output,target],
                                    outputs=o_error)
        dW_i = T.grad(o_error,W_i)
        db_i = T.grad(o_error,b_i)
        dW_o = T.grad(o_error,W_o)
        db_o = T.grad(o_error,b_o)

        self.backward = theano.function(inputs=[input,target,learning_rate],
                                        outputs=None,
                                        updates=[(self.W_i,self.W_i-learning_rate*dW_i),
                                                 (self.b_i,self.b_i-learning_rate*db_i),
                                                 (self.W_o,self.W_o-learning_rate*dW_o),
                                                 (self.b_o,self.b_o-learning_rate*db_o)])
class Gauss:
    def __init__(self):
        self.name = "Gauss"
    def forward(self,r,delta):
        return T.exp(-T.square(r) / (2 * T.square(delta)))
class ReflextedSigmoid:
    def __init__(self):
        self.name = "ReflextedSigmoid"
    def forward(self,r,delta):
        return 1 / (1 + T.exp(np.square(r) / (T.square(delta))))
class InverseMulti:
    def __init__(self):
        self.name = "InverseMulti"
    def forward(self,r,delta):
        return 1/T.sqrt(T.square(r)+T.square(delta))
def select_func(name):
    if name == "Gauss":
        return Gauss
    elif name == "ReflextedSigmoid":
        return ReflextedSigmoid
    elif name == "InverseMulti":
        return InverseMulti
    else:
        raise TypeError("Unknow model:%s"%str(name))
class RBFBPClassification:
    def __init__(self,in_dim,hid_dim,out_dim,act_name = "Gauss"):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        W_i = np.random.rand(in_dim,hid_dim)
        b_i = np.random.rand(hid_dim)
        W_o = np.random.rand(hid_dim,out_dim)
        b_o = np.random.rand(out_dim)
        sigma = np.random.rand(hid_dim)
        mu = np.random.rand(hid_dim)
        self.W_i = theano.shared(value=W_i,name="W_i")
        self.b_i = theano.shared(value=b_i,name="b_i")
        self.W_o = theano.shared(value=W_o,name="W_o")
        self.b_o = theano.shared(value=b_o,name="b_o")
        self.sigma = theano.shared(value=sigma,name="sigma")
        self.mu = theano.shared(value=mu,name="mu")

        self.act_func = select_func(act_name)()

        self.build_model()
    def build_model(self):
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambd")
        W_i = self.W_i
        b_i = self.b_i
        W_o = self.W_o
        b_o = self.b_o
        sigma = self.sigma
        mu = self.mu
        # forward
        y_out = T.dot(input, W_i) + b_i
        hid = self.act_func.forward(y_out-self.mu,self.sigma)
        z_out = T.dot(hid, W_o) + b_o
        output = T.nnet.softmax(z_out)
        self.forward = theano.function(inputs=[input],
                                       outputs=output)
        o_error = T.sum(T.nnet.categorical_crossentropy(output, target))
        length = output.shape[0]
        o_error = o_error / length
        self.loss = theano.function(inputs=[output, target],
                                    outputs=o_error)
        dW_i = T.grad(o_error, W_i)
        db_i = T.grad(o_error, b_i)
        dW_o = T.grad(o_error, W_o)
        db_o = T.grad(o_error, b_o)
        dsigma = T.grad(o_error,sigma)
        dmu = T.grad(o_error,mu)

        self.backward = theano.function(inputs=[input, target, learning_rate,lambd],
                                        outputs=None,
                                        updates=[(self.W_i, self.W_i - learning_rate * (dW_i+lambd*self.W_i)),
                                                 (self.b_i, self.b_i - learning_rate * db_i),
                                                 (self.W_o, self.W_o - learning_rate * (dW_o+lambd*self.W_o)),
                                                 (self.b_o, self.b_o - learning_rate * db_o),
                                                 (self.sigma,self.sigma - learning_rate * dsigma),
                                                 (self.mu,self.mu - learning_rate * dmu)])
class RBFGradClassification:
    def __init__(self,in_dim,out_dim,act_name = "Gauss"):
        self.in_dim = in_dim
        self.out_dim = out_dim

        weight = np.random.rand(in_dim,out_dim)
        sigma = np.random.rand(in_dim)
        mu = np.random.rand(in_dim)
        self.weight = theano.shared(value=weight,name="weight")
        self.sigma = theano.shared(value=sigma,name="sigma")
        self.mu = theano.shared(value=mu,name="mu")

        self.act_func = select_func(act_name)()
        self.bulid_model()
    def bulid_model(self):
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.matrix(name="learning_rate")
        lambd = T.matrix(name="lambda")
        weight = self.weight
        sigma = self.sigma
        mu = self.mu
        # forward

        hid = self.act_func.forward(input - mu,sigma)
        z_out = T.dot(hid,weight)
        output = T.nnet.softmax(z_out)

        self.forward = theano.function(inputs=[input],
                                       outputs=output)
        length = target.shape[0]
        loss_func = T.sum(T.nnet.categorical_crossentropy(output,target))/length

        dW = T.grad(loss_func,weight)
        dsigma = T.grad(loss_func,sigma)
        dmu = T.grad(loss_func,mu)

        self.loss = theano.function(inputs=[output,target],
                                    outputs=loss_func)
        self.backward = theano.function(inputs=[input,target,learning_rate,lambd],
                                        outputs=None,
                                        updates=[(self.weight,self.weight - learning_rate*(dW + lambd*self.weight)),
                                                 (self.sigma,self.sigma - learning_rate*dsigma),
                                                 (self.mu,self.mu - learning_rate*dmu)])
if __name__ == '__main__':
    batch = 5
    in_dim = 10
    input = T.matrix(name="input")
    mu = T.vector(name="mu")
    out = T.sum(input/mu)
    output = T.grad(out,mu)

    fn = theano.function(inputs=[input,mu],
                    outputs=output)
    _mu = np.random.rand(in_dim)
    _input = np.random.rand(batch,in_dim)
    y = fn(_input,_mu)
    print(_mu.shape,_input.shape,y.shape)