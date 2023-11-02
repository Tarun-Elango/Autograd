from src.core import Value
import random

# building a neural net

class Neuron:
  
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
  
  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b] # returns parameter scalars

class Layer:
  
  def __init__(self, nin, nout): # inputs and outputs
    self.neurons = [Neuron(nin) for _ in range(nout)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs # get value out at the last layer
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()] # returns all parameters for all neurons

class MLP:
  
  def __init__(self, nin, nouts): # INPUTS AND list of outputs, nin is input layer, nouts is list of the remaining layers
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()] # returns all parameters for all layers


