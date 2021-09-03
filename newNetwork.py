import numpy as np
import random as rnd
def sigmoid(x):
    # Функция активации sigmoid:: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true и y_pred являются массивами numpy с одинаковой длиной
    return ((y_true - y_pred) ** 2).mean()

class Neuron():
    def __init__(self, weight1, weight2, bias, d_h_d_w1, d_h_d_w2, d_h_d_b):
        self.weight1 = weight1
        self.weight2 = weight2
        self.bias = bias
        self.d_h_d_w1 = d_h_d_w1
        self.d_h_d_w2 = d_h_d_w2
        self.d_h_d_b = d_h_d_b

class OurNeuralNetwork:
    def __init__(self):
        self.qNeurons = 2
        self.weights = self.creatWeights()
        self.bias = self.createBias()
    def creatWeights(self):
        weights = []
        for i in range(3*self.qNeurons):
            weights.append(np.random.normal())
        return weights
    def createBias(self):
        bias = []
        for i in range(self.qNeurons + 1):
            bias.append(np.random.normal())
        return bias
    def feedforward(self, x):
        neurons = [None] * self.qNeurons
        n = 1
        for i in range(self.qNeurons):
            neurons[i] = sigmoid(self.weights[n - 1] * x[0] + self.weights[n] * x[1] + self.bias[i])
            n += 2
        weightsNP = np.array([self.weights[-self.qNeurons:]])
        neuronsNP = np.transpose(np.array([neurons]))
        o1 = sigmoid(np.matmul(weightsNP, neuronsNP)[0][0] + self.bias[-1])
        return o1
    def sumN(self, n, x, i):
        return self.weights[n-1] * x[0] + self.weights[n] * x[1] + self.bias[i]
    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
            Elements in all_y_trues correspond to those in data.
        """
        learn_rate = 0.01
        epochs = 10000  # количество циклов во всём наборе данных
        neurons = [None] * self.qNeurons
        derivResult = [None] * self.qNeurons
        derivWeightOut = [None] * self.qNeurons
        derivNeuronsOut = [None] * self.qNeurons
        neuronsArray = [None] * self.qNeurons

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                n = 1
                k = 1
                for i in range(self.qNeurons):
                    neurons[i] = sigmoid(self.weights[n - 1] * x[0] + self.weights[n] * x[1] + self.bias[i])
                    derivResult[i] = deriv_sigmoid(self.weights[n - 1] * x[0] + self.weights[n] * x[1] + self.bias[i])
                    n += 2

                weightsNP = np.array([self.weights[-self.qNeurons:]])
                neuronsNP = np.transpose(np.array([neurons]))
                o1 = sigmoid(np.matmul(weightsNP, neuronsNP)[0][0] + self.bias[-1])
                derivO1 = deriv_sigmoid(np.matmul(weightsNP, neuronsNP)[0][0] + self.bias[-1])
                y_pred = o1
                d_L_d_ypred = -2 * (y_true - y_pred)

                for i in range(self.qNeurons):
                    derivWeightOut[i] = neurons[i] * derivO1
                    derivNeuronsOut[i] = self.weights[n - 1] * derivO1
                    self.weights[n - 1] -= learn_rate * d_L_d_ypred * derivWeightOut[i]
                    n += 1
                    neuronsArray[i] = Neuron(self.weights[k-1], self.weights[k], self.bias[i], x[0] * deriv_sigmoid(self.sumN(k, x, i)), x[1] * deriv_sigmoid(self.sumN(k, x, i)), deriv_sigmoid(self.sumN(k, x, i)))
                    k += 2
                    neuronsArray[i].weight1 -= learn_rate * d_L_d_ypred * derivNeuronsOut[i] * neuronsArray[i].d_h_d_w1
                    neuronsArray[i].weight2 -= learn_rate * d_L_d_ypred * derivNeuronsOut[i] * neuronsArray[i].d_h_d_w2
                    neuronsArray[i].bias -= learn_rate * d_L_d_ypred * derivNeuronsOut[i] * neuronsArray[i].d_h_d_b
                d_ypred_d_b = derivO1

                self.bias[-1] -= learn_rate * d_L_d_ypred * d_ypred_d_b
                # --- Подсчет частных производных
                # --- Наименование: d_L_d_w1 представляет "частично L / частично w1"

            # --- Подсчитываем общую потерю в конце каждой фазы
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

# Определение набора данных
dataList = []
trueAnsw = []
offsetWeight = 135
offsetHeight = 66
for i in range (10):
    dataList.append([rnd.randint(150 - offsetWeight, 180 - offsetWeight), rnd.randint(65 - offsetHeight, 80 - offsetHeight)]) # Male
    trueAnsw.append(0)
    dataList.append([rnd.randint(120 - offsetWeight, 134 - offsetWeight), rnd.randint(60 - offsetHeight, 64 - offsetHeight)]) # Female
    trueAnsw.append(1)
data = np.array(dataList)

all_y_trues = np.array(trueAnsw)

# Тренируем нашу нейронную сеть!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
emily = np.array([-7, -3])  # 128 фунтов, 63 дюйма
frank = np.array([20, 2])  # 155 фунтов, 68 дюймов
print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank))  # 0.039 - M
#print(dataList, trueAnsw)