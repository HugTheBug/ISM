from functools import reduce
import matplotlib.pyplot as plt

class LinearCongruentialGenerator:
	def __init__(self, a = 16387, betta = 16387, c = 0, M = 2 ** 31):
		self.a = a
		self.betta = betta
		self.c = c
		self.M = M

	def next(self):
		self.a = (self.a * self.betta + self.c) % self.M
		return self.a / self.M

class GeneratorGenerator:
	def __init__(self):
		self.M = 2 ** 31
		self.generator = LinearCongruentialGenerator()

	def next(self):
		return LinearCongruentialGenerator(
			self.M * self.generator.next(),
			self.M * self.generator.next(),
			0,
			self.M)

globalGenerator = GeneratorGenerator()

def isNormalized(A):
    # const isNormalized1 = (A) => A.reduce((ac, str, i) =>
    # ac + str.reduce((acc, v, ind) => acc + (v - (i === ind)) ** 2, 0), 0) < 1;

    value = 0.0
    for i in range(len(A)):
        for j in range(len(A)):
            value += (A[i][j] - (i == j)) ** 2
    if value < 1:
        return True

    max_value = 0
    for i in range(len(A)):
        value = 0
        for j in range(len(A)):
            value += abs(A[i][j] - (i == j))
        if value > max_value:
            max_value = value
    return max_value < 1

def solve(A, b, N = 10):
    max_iterations = 50000

    if not isNormalized(A):
        raise NameError("normalize it or smth")

    # Ax = f -> x = Ax + f
    A = [[-elem for elem in row] for row in A]
    for i in range(len(A)):
        A[i][i] += 1
    
    pi = [1 / len(A)] * len(A)
    p = [[1 / len(A)] * len(A) for _ in range(len(A))]
    random = globalGenerator.next()

    x = [0] * len(A)
    for j in range(max_iterations):
        i = [0] * N
        for k in range(N):
            rand = random.next()
            for z in range(len(pi)):
                if rand < pi[z]:
                    i[k] = z
                    break
                rand -= pi[z]

        Q = [0] * N
        ind = i[0]
        if pi[i[0]] > 0:
            Q[0] = 1 / pi[i[0]]
        else:
            Q[0] = 0
        for m in range(1, N):
            if p[i[m - 1]][i[m]] > 0:
                Q[m] = Q[m - 1] * A[i[m - 1]][i[m]] / p[i[m - 1]][i[m]]
            else:
                Q[m] = 0
        ksiN = 0
        for m in range(N):
            ksiN += b[i[m]] * Q[m]
        x[ind] += ksiN
    return [elem / max_iterations for elem in x]

def getDiscrepancy(A, x, b):
    discrepancy = []
    for i in range(len(b)):
        value = 0
        for j in range(len(x)):
            value += A[i][j] * x[j]
        discrepancy.append(value - b[i])
    return discrepancy
    
def get_point(A, b):
    x = []
    y = []
    for i in range(1, 20):
        x.append(i)
        y.append(max(getDiscrepancy(A, solve(A, b, i), b), key=abs))
    return x, y

def main():
    A = [[1.2, 0.1, -0.3],
        [-0.3, 0.9, -0.2],
        [0.4, 0.5, 1.0]]
    b = [2, 3, 3]

    x, y = get_point(A, b)
    plt.plot(x, y, c='b')
    plt.plot(x, [0 for _ in range(len(x))], c='r')    

    plt.show()
main()