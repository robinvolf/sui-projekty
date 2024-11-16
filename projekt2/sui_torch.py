import numpy as np


class Tensor:
    # Rozhraní back_op: trojice (operace, zdroj1, [zdroj2]), kde zdroj2 nemají všechny operace.
    # Zdroje jsou tenzory, z nichž tento tenzor vznikl, operace je prostě název funkce,
    # která byla zavolána nad dvěma zdrojovými tenzory a výsledkem byl tento tenzor
    def __init__(self, value, back_op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.back_op = back_op

    def __str__(self):
        str_val = str(self.value)
        str_val = '\t' + '\n\t'.join(str_val.split('\n'))
        str_bwd = str(self.back_op.__class__.__name__)
        return 'Tensor(\n' + str_val + '\n\tbwd: ' + str_bwd + '\n)'

    @property
    def shape(self):
        return self.value.shape
    
    # 1. Spočítám nový gradient aktuálního tenzoru podle toho, z jaké vznikl operace
    # 2. Pokud předcházející uzel ve výpočetním grafu (zdrojový tenzor) není listový (nemá back_op = None), zavolám na něj backward() s mým novým gradientem
    def backward(self, deltas=None):
        # Inicializuje deltas, pokud nebyly dodány
        if deltas is None:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(f"Can only backpropagate a scalar, got shape {self.shape}")
        
            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')
            
            deltas = np.ones_like(self.value)  # Skalární gradient

        # Akumulace gradientů
        if self.grad is None:
            self.grad = deltas
        else:
            self.grad += deltas

        # Pokud jsme uzel, nepropagujeme dál
        if self.back_op is not None:
            match self.back_op[0]:
                case "add":
                    partial_derivatives = np.multiply(np.ones_like(self.value), deltas)
                    self.back_op[1].backward(partial_derivatives)
                    self.back_op[2].backward(partial_derivatives)

                case "subtract":
                    self.back_op[1].backward(deltas)
                    self.back_op[2].backward(deltas * -1)

                case "multiply":
                    self.back_op[1].backward(deltas * self.back_op[2].value)
                    self.back_op[2].backward(deltas * self.back_op[1].value)

                case "relu":
                    relu_grad = np.where(self.back_op[1].value > 0, 1, 0)
                    self.back_op[1].backward(deltas * relu_grad)

                case "dot_product":
                    if self.back_op[2] is None:
                        raise ValueError('Tensor created from dot product must have two parents')
                    self.back_op[1].backward(np.dot(deltas, self.back_op[2].value.T))
                    self.back_op[2].backward(np.dot(self.back_op[1].value.T, deltas))

                case "sui_sum":
                    grad = np.ones_like(self.back_op[1].value) * deltas  # Nastavení gradientu všech prvků na 1
                    self.back_op[1].grad += grad  # Akumulace gradientu
                    
                    if self.back_op[1].back_op is not None:
                        self.back_op[1].backward(deltas=grad)

                case _:
                    raise NotImplementedError(f"Operation '{self.back_op[0]}' not implemented")
    
def sui_sum(tensor):
    new_value = np.sum(tensor.value)
    return Tensor(new_value, ("sui_sum", tensor))

def add(a, b):
    new_value = a.value + b.value
    return Tensor(new_value, ("add", a, b))

def subtract(a, b):
    new_value = a.value - b.value
    return Tensor(new_value, ("subtract", a, b))

def multiply(a, b):
    new_value = a.value * b.value
    return Tensor(new_value, ("multiply", a, b))

def relu(tensor):
    new_value = np.maximum(tensor.value, 0)
    return Tensor(new_value, ("relu", tensor))

def dot_product(a, b):
    new_value = np.dot(a.value, b.value)
    return Tensor(new_value, ("dot_product", a, b))
