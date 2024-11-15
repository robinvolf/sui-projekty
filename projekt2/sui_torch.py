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
        if deltas is not None:
            assert deltas.shape == self.value.shape, f'Expected gradient with shape {self.value.shape}, got {deltas.shape}'

            self.grad = deltas

            # Pokud jsme uzel, nepropagujeme dál
            if self.back_op is not None:
                match self.back_op[0]:
                    case "add":
                        #assert(self.back_op[2], 'Tensor created from addition must have two parents')      # Caused warning
                        if self.back_op[2] is None:
                            raise ValueError('Tensor created from addition must have two parents')
                        # Protože, když zderivuju součet, parciální derivace je 1
                        # Ještě vynásobím gradientem, co jsem dostal
                        partial_derivatives = np.multiply(np.ones_like(self.value), deltas)
                        self.back_op[1].grad += partial_derivatives
                        self.back_op[2].grad += partial_derivatives
                        self.back_op[1].backward(partial_derivatives)
                        self.back_op[2].backward(partial_derivatives)
                    
                    case "subtract":
                        #assert(self.back_op[2], 'Tensor created from subtraction must have two parents')      # Caused warning
                        if self.back_op[2] is None:
                            raise ValueError('Tensor created from subtraction must have two parents')
                        # Protože, když zderivuju rozdíl parciální derivace je 1 pro první složku a -1 pro druhou (protože ji odečítám)
                        self.back_op[1].grad += deltas
                        self.back_op[2].grad += deltas * -1
                        self.back_op[1].backward(deltas)
                        self.back_op[2].backward(deltas * -1)
                    
                    case "multiply":
                        #assert(self.back_op[2], 'Tensor created from subtraction must have two parents')
                        if self.back_op[2] is None:
                            raise ValueError('Tensor created from multiplication must have two parents')
                        # dL/da = b, dL/db = a
                        self.back_op[1].grad += deltas * self.back_op[2].value
                        self.back_op[2].grad += deltas * self.back_op[1].value
                        self.back_op[1].backward(deltas * self.back_op[2].value)
                        self.back_op[2].backward(deltas * self.back_op[1].value)
                    
                    case "relu":
                        if self.back_op[1] is None:
                            raise ValueError('Tensor created from relu must have one parents')
                        # Derivative of ReLU is 1 for positive values and 0 for negative
                        relu_grad = np.where(self.back_op[1].value > 0, 1, 0)
                        self.back_op[1].grad += deltas * relu_grad
                        self.back_op[1].backward(deltas * relu_grad)
                    
                    case "dot_product":
                        # dL/da = b, dL/db = a
                        if self.back_op[2] is None:
                            raise ValueError('Tensor created from dot product must have two parents')
                        self.back_op[1].grad += np.dot(deltas, self.back_op[2].value.T)
                        self.back_op[2].grad += np.dot(self.back_op[1].value.T, deltas)
                        self.back_op[1].backward(np.dot(deltas, self.back_op[2].value.T))
                        self.back_op[2].backward(np.dot(self.back_op[1].value.T, deltas))
        else:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(f'Can only backpropagate a scalar, got shape {self.shape}')

            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')

            match self.back_op[0]:
                case "sui_sum":
                    # Protože když zderivuju a_1 + a_2 + a_3 ... + a_n podle a_i dostanu vektor (1, 1, 1, ... 1)
                    self.back_op[1].grad = np.ones_like(self.back_op[1].value)
                    if self.back_op[1].back_op is not None:
                        new_deltas = self.back_op[1].grad
                        self.back_op[1].backward(deltas=new_deltas)
                case _:
                    raise NotImplementedError(f'Unimplemented source operation: {self.back_op[0]}')

# Jak na to?
# Každá operace vrátí nový Tenzor, který budu výsledkem operace
# Tento nový tenzor bude mít nastavený atribut back_op, abychom pak byli schopni se vracet ve výpočetním grafu a šířit zpětně chybu (metoda backward()).
#
# Jsme už tak low-level, že se na to dívám jako na výpočetní graf a ne jako na neuronku. To za mě řeší ve funkcích ze zadání.

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
