from src.core import Value
from src.nn import Neuron, Layer, MLP

# nn architecture 
x = [2.0, 3.0, -1.0]
n = MLP(3, [4,4,1]) # 2 inputs and 3 output, output is 3 evalutions of the 3 neurons
n(x)

#dataset
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
] # 4 possible inputs
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

for i in range(20):
    #gradient descent, follow gradient info and minimize loss    

    ypred = [n(x) for x in xs] # forward 
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)]) # find loss

    for p in n.parameters(): # all parameters grad is set to zero, casue when we do second backward it goes on top of that
         p.data

    loss.backward() # backward pass

    for p in n.parameters(): # nudge data 
    # update p data according to gradient
        p.data += -0.01*p.grad # minimize loss, i.e increase data decrease loss
    print(i, loss.data)