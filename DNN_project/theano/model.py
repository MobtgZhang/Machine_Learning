import theano
import theano.tensor as T
import numpy as np

class Linear:
    def __init__(self,in_dim,out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        W = np.random.uniform(-np.sqrt(1.0/in_dim),np.sqrt(1.0/in_dim),(in_dim,out_dim))
        b = np.random.uniform(-np.sqrt(1.0/out_dim),np.sqrt(1.0/out_dim),(out_dim))

        self.weight = theano.shared(value=W.astype(theano.config.floatX),name="weight")
        self.bais = theano.shared(value=b.astype(theano.config.floatX),name="bais")
        self.gradient = {}
    def forwad(self,x):
        W_plus_b = T.dot(x,self.weight) + self.bais
        return W_plus_b
    def update_grad(self,loss_func):
        gW,gb = T.grad(loss_func,[self.weight,self.bais])
        self.gradient["grad_weight"] = gW
        self.gradient["grad_bais"] = gb
    def size(self):
        return (self.in_dim,self.out_dim)
class DNNNet:
    def __init__(self,dim_list,act_name,name):
        self.dim_list = dim_list
        num_layers = len(dim_list) - 1
        self.hid_layers_list = []
        self.act_name = act_name
        self.name = name
        for k in range(num_layers):
            fc = Linear(dim_list[k],dim_list[k+1])
            self.hid_layers_list.append(fc)
        self._build_model()
    def _build_model(self):
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambda")
        # defination of predictions
        predict = input
        for k in range(len(self.hid_layers_list)):
            if self.act_name == "sigmoid":
                hidden = T.nnet.sigmoid(predict)
            elif self.act_name == "relu":
                hidden = T.nnet.relu(predict,alpha=0.1)
            elif self.act_name == "tanh":
                hidden = T.tanh(predict)
            elif self.act_name == "softplus":
                hidden = T.nnet.softplus(predict)
            else:
                raise TypeError("Unknown function model: %s"%str(self.act_name))
            predict = self.hid_layers_list[k].forwad(hidden)
        # defination of loss functions
        delta = target - predict
        batch = delta.shape[0]
        length = delta.shape[1]
        loss_func = T.sum(T.sum(T.square(target-predict),axis=0)/length)/batch
        # get loss
        self.loss = theano.function(inputs=[target,predict],
                                    outputs=loss_func)
        # forward
        self.forward = theano.function(inputs=[input],
                                       outputs=predict)
        # backward
        for k in range(len(self.hid_layers_list)):
            self.hid_layers_list[k].update_grad(loss_func)
        # training processing
        updates_list = []
        for k in range(len(self.hid_layers_list)):
            W = self.hid_layers_list[k].weight
            b = self.hid_layers_list[k].bais
            gW = self.hid_layers_list[k].gradient["grad_weight"]
            gb = self.hid_layers_list[k].gradient['grad_bais']
            updates_list.append((W,W-learning_rate*(gW+lambd*W)))
            updates_list.append((b,b-learning_rate*gb))
        self.train = theano.function(inputs=[input,target,learning_rate,lambd],
                        outputs=loss_func,
                        updates=updates_list)
    def save_dict_parameters(self,filename):
        weight_list = []
        bais_list = []
        act_name =self.act_name
        name = self.name
        for k in range(len(self.hid_layers_list)):
            weight_list.append(self.hid_layers_list[k].weight.get_value())
            bais_list.append(self.hid_layers_list[k].bais.get_value())
        np.savez(filename,weight_list=weight_list,bais_list=bais_list,act_name = act_name,name = name)
    @staticmethod
    def load_dict_parameters(filename):
        outfile = np.load(filename)
        weight_list = outfile["weight_list"]
        bais_list = outfile["bais_list"]
        act_name = outfile["act_name"]
        name = outfile["name"]
        dim_list = []
        for k in range(len(weight_list)):
            dim_list.append(weight_list[k].shape[0])
        dim_list.append(weight_list[-1][0])
        model = DNNNet(dim_list,act_name,name)
        for k in range(len(model.hid_layers_list)):
            model.hid_layers_list[k].weight.set_value(weight_list[k])
            model.hid_layers_list[k].bais.set_value(bais_list[k])
        return model
def main():
    batch = 10
    in_dim = 8
    out_dim = 5
    dim_list = [in_dim,10,20,50,30,out_dim]
    input = np.random.rand(batch,in_dim)
    target = np.random.rand(batch,out_dim)
    dnnnet = DNNNet(dim_list,"relu")
    #output = dnnnet.forward(input)

    # print(input.shape,output.shape)
    # dnnnet.loss(input,target)
if __name__ == '__main__':
    main()