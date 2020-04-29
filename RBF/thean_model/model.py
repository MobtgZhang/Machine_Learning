import numpy as np
import theano
import theano.tensor as T

from util import Gauss,Sigmoid,Inverse

class BPClassification:
    def __init__(self,in_dim,hid_dim,n_class,act_name,name=None):
        self.in_size = in_dim
        self.hid_dim = hid_dim
        self.n_class = n_class
        self.act_name = act_name
        self.name = name

        Wi = np.random.rand(in_dim,hid_dim)
        bi = np.random.rand(hid_dim,)
        Wo = np.random.rand(hid_dim,n_class)
        bo = np.random.rand(n_class,)
        self.Wi = theano.shared(value=Wi,name="Wi")
        self.bi = theano.shared(value=bi,name="bi")
        self.Wo = theano.shared(value=Wo,name="Wo")
        self.bo = theano.shared(value=bo,name="bo")
        self.build_theano()
    def build_theano(self):
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambda")
        # forward process
        if self.act_name == "sigmoid":
            hid = T.nnet.sigmoid(T.dot(input,self.Wi)+self.bi)
        elif self.act_name == "relu":
            hid = T.nnet.relu(T.dot(input,self.Wi)+self.bi,0)
        elif self.act_name == "softplus":
            hid = T.nnet.softplus(T.dot(input,self.Wi)+self.bi)
        else:
            raise TypeError("Unknown activate function %s"%str(self.act_name))
        output = T.nnet.softmax(T.dot(hid,self.Wo)+self.bo)

        self.forward = theano.function(inputs=[input],
                                       outputs=output)
        self.predict = theano.function(inputs= [input],
                                       outputs=T.argmax(output,axis=1))
        #loss function
        loss = T.mean(T.nnet.categorical_crossentropy(output,target))
        self.loss = theano.function(inputs=[output,target],
                                    outputs=loss)
        #gradients
        dWi = T.grad(loss,self.Wi)
        dbi = T.grad(loss,self.bi)
        dWo = T.grad(loss,self.Wo)
        dbo = T.grad(loss,self.bo)
        # update gradients
        self.sgd_backward = theano.function(inputs=[input,target,learning_rate,lambd],
                                            outputs=None,
                                            updates=[(self.Wi,self.Wi-learning_rate*(dWi+lambd*self.Wi)),
                                                     (self.bi,self.bi-learning_rate*dbi),
                                                     (self.Wo,self.Wo-learning_rate*(dWo+lambd*self.Wo)),
                                                     (self.bo,self.bo-learning_rate*dbo)])
    def save_parameters(self,filename):
        Wi = self.Wi.get_value()
        bi = self.bi.get_value()
        Wo = self.Wo.get_value()
        bo = self.bo.get_value()
        act_name = self.act_name
        name = self.name
        np.savez(filename,Wi = Wi,bi = bi,Wo=Wo,bo=bo,act_name=act_name,name=name)
    @staticmethod
    def load_parameters(filename):
        npzfile = np.load(filename)
        Wi = npzfile["Wi"]
        bi = npzfile["bi"]
        Wo = npzfile["Wo"]
        bo = npzfile["bo"]
        act_name = npzfile["act_name"]
        name = npzfile["name"]
        in_dim = Wi.shape[0]
        hid_dim = Wi.shape[1]
        n_class = Wo.shape[1]
        model = BPClassification(in_dim,hid_dim,n_class,act_name,name)
        model.Wi.set_value(Wi)
        model.bi.set_value(bi)
        model.Wo.set_value(Wo)
        model.bo.set_value(bo)
        return model
class BPRegression:
    def __init__(self,in_dim,hid_dim,out_dim,act_name,name=None):
        self.in_size = in_dim
        self.hid_size = hid_dim
        self.out_size = out_dim
        self.act_name = act_name
        self.name = name

        Wi = np.random.rand(in_dim,hid_dim)
        bi = np.random.rand(hid_dim,)
        Wo = np.random.rand(hid_dim,out_dim)
        bo = np.random.rand(out_dim,)
        self.Wi = theano.shared(value=Wi,name="Wi")
        self.bi = theano.shared(value=bi,name="bi")
        self.Wo = theano.shared(value=Wo,name="Wo")
        self.bo = theano.shared(value=bo,name="bo")
        self.build_theano()
    def build_theano(self):
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambda")
        # forward process
        if self.act_name == "sigmoid":
            hid = T.nnet.sigmoid(T.dot(input, self.Wi) + self.bi)
        elif self.act_name == "relu":
            hid = T.nnet.relu(T.dot(input, self.Wi) + self.bi, 0)
        elif self.act_name == "softplus":
            hid = T.nnet.softplus(T.dot(input, self.Wi) + self.bi)
        else:
            raise TypeError("Unknown activate function %s" % str(self.act_name))
        output = T.dot(hid, self.Wo) + self.bo

        self.forward = theano.function(inputs=[input],
                                       outputs=output)
        # loss function
        loss = T.mean(T.square(target-output))
        self.loss = theano.function(inputs=[output, target],
                                    outputs=loss)
        # gradients
        dWi = T.grad(loss, self.Wi)
        dbi = T.grad(loss, self.bi)
        dWo = T.grad(loss, self.Wo)
        dbo = T.grad(loss, self.bo)
        # update gradients
        self.sgd_backward = theano.function(inputs=[input, target, learning_rate, lambd],
                                            outputs=None,
                                            updates=[(self.Wi, self.Wi - learning_rate * (dWi + lambd * self.Wi)),
                                                     (self.bi, self.bi - learning_rate * dbi),
                                                     (self.Wo, self.Wo - learning_rate * (dWo + lambd * self.Wo)),
                                                     (self.bo, self.bo - learning_rate * dbo)])
    def save_parameters(self,filename):
        Wi = self.Wi.get_value()
        bi = self.bi.get_value()
        Wo = self.Wo.get_value()
        bo = self.bo.get_value()
        act_name = self.act_name
        name = self.name
        np.savez(filename, Wi=Wi, bi=bi, Wo=Wo, bo=bo, act_name=act_name, name=name)
    @staticmethod
    def load_parameters(filename):
        npzfile = np.load(filename)
        Wi = npzfile["Wi"]
        bi = npzfile["bi"]
        Wo = npzfile["Wo"]
        bo = npzfile["bo"]
        act_name = npzfile["act_name"]
        name = npzfile["name"]
        in_dim = Wi.shape[0]
        hid_dim = Wi.shape[1]
        n_class = Wo.shape[1]
        model = BPRegression(in_dim, hid_dim, n_class, act_name, name)
        model.Wi.set_value(Wi)
        model.bi.set_value(bi)
        model.Wo.set_value(Wo)
        model.bo.set_value(bo)
        return model
class RBFGradClassification:
    def __init__(self,in_dim,n_class,act_name,name=None):
        self.in_dim = in_dim
        self.n_class = n_class
        self.act_name = act_name
        self.name = name
        weight = np.random.rand(in_dim,n_class)
        self.weight = theano.shared(value=weight,name="weight")
        mu = np.random.rand(in_dim,)
        self.mu = theano.shared(value=mu,name="mu")
        gamma = np.random.rand(in_dim,)
        self.gamma = theano.shared(value=gamma,name="gamma")
        self.build_model()
    def build_model(self):
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambda")
        # forward process
        if self.act_name == "gauss":
            hid = Gauss(input,self.mu,self.gamma)
        elif self.act_name == "sigmoid":
            hid = Sigmoid(input,self.mu,self.gamma)
        elif self.act_name == "inverse":
            hid = Inverse(input,self.mu,self.gamma)
        else:
            raise TypeError("Unknown model type: %s"%self.act_name)
        hid = T.dot(hid,self.weight)
        output = T.nnet.softmax(hid)
        self.forward = theano.function(inputs=[input],
                                       outputs=output)
        self.predict = theano.function(inputs=[input],
                                       outputs=T.argmax(output,axis=1))
        # loss function
        loss = T.mean(T.nnet.categorical_crossentropy(output, target))
        self.loss = theano.function(inputs=[output, target],
                                    outputs=loss)
        # gradients
        dweight = T.grad(loss,self.weight)
        dmu = T.grad(loss,self.mu)
        dgamma = T.grad(loss,self.gamma)
        # update gradients
        self.sgd_backward = theano.function(inputs=[input,target,learning_rate,lambd],
                                            outputs=None,
                                            updates=[(self.weight,self.weight-learning_rate*(dweight+lambd*self.weight)),
                                                     (self.mu,self.mu -learning_rate*dmu),
                                                     (self.gamma,self.gamma-learning_rate*dgamma)])
    def save_parameters(self,filename):
        weight = self.weight.get_value()
        mu = self.mu.get_value()
        gamma = self.gamma.get_value()
        act_name = self.act_name
        name = self.name
        np.savez(filename, weight=weight,mu=mu,gamma=gamma,act_name=act_name, name=name)
    @staticmethod
    def load_parameters(self,filename):
        npzfile = np.load(filename)
        weight = npzfile["weight"]
        mu = npzfile["mu"]
        gamma = npzfile["gamma"]
        act_name = npzfile["act_name"]
        name = npzfile["name"]
        in_dim = weight.shape[0]
        n_class = weight.shape[1]
        model = RBFGradClassification(in_dim, n_class, act_name, name)
        model.weight.set_value(weight)
        model.mu.set_value(mu)
        model.gamma.set_value(gamma)
        return model
class RBFGradRegression:
    def __init__(self,in_dim,out_dim,act_name,name=None):
        self.in_dim = in_dim
        self.n_class = out_dim
        self.act_name = act_name
        self.name = name
        weight = np.random.rand(in_dim,out_dim)
        self.weight = theano.shared(value=weight,name="weight")
        mu = np.random.rand(in_dim,)
        self.mu = theano.shared(value=mu,name="mu")
        gamma = np.random.rand(in_dim,)
        self.gamma = theano.shared(value=gamma,name="gamma")
        self.build_model()
    def build_model(self):
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambda")
        # forward process
        if self.act_name == "gauss":
            hid = Gauss(input, self.mu, self.gamma)
        elif self.act_name == "sigmoid":
            hid = Sigmoid(input, self.mu, self.gamma)
        elif self.act_name == "inverse":
            hid = Inverse(input, self.mu, self.gamma)
        else:
            raise TypeError("Unknown model type: %s" % self.act_name)
        output = T.dot(hid, self.weight)
        self.forward = theano.function(inputs=[input],
                                       outputs=output)
        # loss function
        loss = T.mean(T.square(target - output))
        self.loss = theano.function(inputs=[output, target],
                                    outputs=loss)
        # gradients
        dweight = T.grad(loss,self.weight)
        dmu = T.grad(loss,self.mu)
        dgamma = T.grad(loss,self.gamma)
        # update gradients
        self.sgd_backward = theano.function(inputs=[input, target, learning_rate, lambd],
                                            outputs=None,
                                            updates=[(self.weight,
                                                      self.weight - learning_rate * (dweight + lambd * self.weight)),
                                                     (self.mu, self.mu - learning_rate * dmu),
                                                     (self.gamma, self.gamma - learning_rate * dgamma)])
    def save_parameters(self,filename):
        weight = self.weight.get_value()
        mu = self.mu.get_value()
        gamma = self.gamma.get_value()
        act_name = self.act_name
        name = self.name
        np.savez(filename, weight=weight, mu=mu, gamma=gamma, act_name=act_name, name=name)
    def load_parameters(self,filename):
        npzfile = np.load(filename)
        weight = npzfile["weight"]
        mu = npzfile["mu"]
        gamma = npzfile["gamma"]
        act_name = npzfile["act_name"]
        name = npzfile["name"]
        in_dim = weight.shape[0]
        n_class = weight.shape[1]
        model = RBFGradRegression(in_dim, n_class, act_name, name)
        model.weight.set_value(weight)
        model.mu.set_value(mu)
        model.gamma.set_value(gamma)
        return model
class RBFBPClassification:
    def __init__(self,in_dim,hid_dim,n_class,act_name,name = None):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = n_class
        self.act_name = act_name
        self.name = name

        Wi = np.random.rand(in_dim,hid_dim)
        bi = np.random.rand(hid_dim,)
        Wo = np.random.rand(hid_dim,n_class)
        bo = np.random.rand(n_class,)

        self.Wi = theano.shared(value=Wi,name="Wi")
        self.bi = theano.shared(value=bi,name="bi")
        self.Wo = theano.shared(value=Wo,name="Wo")
        self.bo = theano.shared(value=bo,name="bo")

        hmu = np.random.rand(hid_dim,)
        hgamma = np.random.rand(hid_dim,)
        self.hmu = theano.shared(value=hmu,name="hmu")
        self.hgamma = theano.shared(value=hgamma,name="hgamma")
        self.build_model()
    def build_model(self):
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambda")
        #forward
        tmp = T.dot(input, self.Wi) + self.bi
        if self.act_name == "gauss":
            hid = Gauss(tmp,self.hmu,self.hgamma)
        elif self.act_name == "sigmoid":
            hid = Sigmoid(tmp,self.hmu,self.hgamma)
        elif self.act_name == "inverse":
            hid = Inverse(tmp,self.hmu,self.hgamma)
        else:
            raise TypeError("Unknown model type:%s"%str(self.act_name))
        output = T.nnet.softmax(T.dot(hid,self.Wo)+self.bo)
        self.forward = theano.function(inputs=[input],
                                       outputs=output)
        self.predict = theano.function(inputs=[input],
                                       outputs=T.argmax(output, axis=1))
        # loss function
        loss = T.mean(T.nnet.categorical_crossentropy(output, target))
        self.loss = theano.function(inputs=[output, target],
                                    outputs=loss)
        # gradients
        dWi = T.grad(loss,self.Wi)
        dbi = T.grad(loss,self.bi)
        dWo = T.grad(loss,self.Wo)
        dbo = T.grad(loss,self.bo)

        dhmu = T.grad(loss,self.hmu)
        dhgamma = T.grad(loss,self.hgamma)

        # update gradients
        self.sgd_backward = theano.function(inputs=[input, target, learning_rate, lambd],
                                            outputs=None,
                                            updates=[(self.Wi,self.Wi - learning_rate * (dWi + lambd * self.Wi)),
                                                     (self.bi, self.bi - learning_rate * dbi),
                                                     (self.Wo, self.Wo - learning_rate * (dWo + lambd * self.Wo)),
                                                     (self.bo, self.bo - learning_rate * dbo),
                                                     (self.hmu, self.hmu - learning_rate * dhmu),
                                                     (self.hgamma, self.hgamma - learning_rate * dhgamma)])

    def save_parameters(self, filename):
        Wi = self.Wi.get_value()
        bi = self.bi.get_value()
        Wo = self.Wo.get_value()
        bo = self.bo.get_value()
        hmu = self.hmu.get_value()
        hgamma = self.hgamma.get_value()
        act_name = self.act_name
        name = self.name
        np.savez(filename, Wi=Wi, bi=bi, Wo=Wo, bo=bo,
                 hmu=hmu,hgamma=hgamma,act_name=act_name, name=name)

    @staticmethod
    def load_parameters(filename):
        npzfile = np.load(filename)
        Wi = npzfile["Wi"]
        bi = npzfile["bi"]
        Wo = npzfile["Wo"]
        bo = npzfile["bo"]
        hmu = npzfile["hmu"]
        hgamma = npzfile["hgamma"]
        act_name = npzfile["act_name"]
        name = npzfile["name"]
        in_dim = Wi.shape[0]
        hid_dim = Wi.shape[1]
        n_class = Wo.shape[1]
        model = RBFBPClassification(in_dim, hid_dim, n_class, act_name, name)
        model.Wi.set_value(Wi)
        model.bi.set_value(bi)
        model.Wo.set_value(Wo)
        model.bo.set_value(bo)
        model.hmu.set_value(hmu)
        model.hgamma.set_value(hgamma)
        return model
class RBFBPRegression:
    def __init__(self,in_dim,hid_dim,out_dim,act_name,name):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.act_name = act_name
        self.name = name

        Wi = np.random.rand(in_dim,hid_dim)
        bi = np.random.rand(hid_dim,)
        Wo = np.random.rand(hid_dim,out_dim)
        bo = np.random.rand(out_dim,)

        self.Wi = theano.shared(value=Wi,name="Wi")
        self.bi = theano.shared(value=bi,name="bi")
        self.Wo = theano.shared(value=Wo,name="Wo")
        self.bo = theano.shared(value=bo,name="bo")
        hmu = np.random.rand(hid_dim, )
        hgamma = np.random.rand(hid_dim, )
        self.hmu = theano.shared(value=hmu, name="hmu")
        self.hgamma = theano.shared(value=hgamma, name="hgamma")
        self.build_model()

    def build_model(self):
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambda")
        # forward process
        tmp = T.dot(input, self.Wi) + self.bi
        if self.act_name == "gauss":
            hid = Gauss(tmp, self.hmu, self.hgamma)
        elif self.act_name == "sigmoid":
            hid = Sigmoid(tmp, self.hmu, self.hgamma)
        elif self.act_name == "inverse":
            hid = Inverse(tmp, self.hmu, self.hgamma)
        else:
            raise TypeError("Unknown model type:%s" % str(self.act_name))
        output = T.dot(hid, self.Wo) + self.bo
        self.forward = theano.function(inputs=[input],
                                       outputs=output)
        # loss function
        loss = T.mean(T.square(target - output))
        self.loss = theano.function(inputs=[output, target],
                                    outputs=loss)
        # gradients
        dWi = T.grad(loss, self.Wi)
        dbi = T.grad(loss, self.bi)
        dWo = T.grad(loss, self.Wo)
        dbo = T.grad(loss, self.bo)

        dhmu = T.grad(loss, self.hmu)
        dhgamma = T.grad(loss, self.hgamma)

        # update gradients
        self.sgd_backward = theano.function(inputs=[input, target, learning_rate, lambd],
                                            outputs=None,
                                            updates=[(self.Wi, self.Wi - learning_rate * (dWi + lambd * self.Wi)),
                                                     (self.bi, self.bi - learning_rate * dbi),
                                                     (self.Wo, self.Wo - learning_rate * (dWo + lambd * self.Wo)),
                                                     (self.bo, self.bo - learning_rate * dbo),
                                                     (self.hmu, self.hmu - learning_rate * dhmu),
                                                     (self.hgamma, self.hgamma - learning_rate * dhgamma)])

    def save_parameters(self, filename):
        Wi = self.Wi.get_value()
        bi = self.bi.get_value()
        Wo = self.Wo.get_value()
        bo = self.bo.get_value()
        hmu = self.hmu.get_value()
        hgamma = self.hgamma.get_value()
        act_name = self.act_name
        name = self.name
        np.savez(filename, Wi=Wi, bi=bi, Wo=Wo, bo=bo,
                 hmu=hmu, hgamma=hgamma, act_name=act_name, name=name)

    @staticmethod
    def load_parameters(filename):
        npzfile = np.load(filename)
        Wi = npzfile["Wi"]
        bi = npzfile["bi"]
        Wo = npzfile["Wo"]
        bo = npzfile["bo"]
        hmu = npzfile["hmu"]
        hgamma = npzfile["hgamma"]
        act_name = npzfile["act_name"]
        name = npzfile["name"]
        in_dim = Wi.shape[0]
        hid_dim = Wi.shape[1]
        n_class = Wo.shape[1]
        model = RBFBPRegression(in_dim, hid_dim, n_class, act_name, name)
        model.Wi.set_value(Wi)
        model.bi.set_value(bi)
        model.Wo.set_value(Wo)
        model.bo.set_value(bo)
        model.hmu.set_value(hmu)
        model.hgamma.set_value(hgamma)
        return model
