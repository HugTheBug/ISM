from math import exp, log, inf, tan, pi, gamma, sqrt
from numpy import prod
import matplotlib.pyplot as plt
from functools import reduce

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

class NormalGenerator:
	def __init__(self, m = 0, sSquared = 1, n = 64):
		self.m = 0
		self.s = sqrt(sSquared)
		self.generators = []
		self.n = 64

		for _ in range(n):
			self.generators.append(globalGenerator.next())

	def next(self):
		return self.m + self.s * sqrt(12 / self.n) * (reduce(lambda ac, el: ac + el.next(), self.generators, 0) - self.n / 2)

class ChiSquaredGenerator:
	def __init__(self, n = 64):
		self.generators = []

		for _ in range(n):
			self.generators.append(NormalGenerator())

	def next(self):
		return reduce(lambda ac, el: ac + el.next() ** 2, self.generators, 0)

class UniformGenerator:
	def __init__(self, a, b):
		self.a = a
		self.b = b
		self.generator = globalGenerator.next()

	def next(self):
		return self.a + self.generator.next() * (self.b - self.a)

def normalDensity(m, sSquare, x):
	return 1 / sqrt(sSquare * 2 * pi) * exp(-((x - m) ** 2 / (2 * sSquare)))

def chiSquaredDensity(k, x):
	return 0.5 ** (k / 2) / gamma(k / 2) * x ** (k / 2 - 1) * exp(-x / 2)

def uniformDensity(a, b, x):
	if x >= a and x <= b:
		return 1 / (b - a)
	else:
		return 0 

def getExpectedValue(x):
  return sum(x) / len(x)

def getMonteCarloValue(low, high):
	if low == -inf and high == inf:
		return NormalGenerator(0, 100).next()
	elif high == inf and low != -inf: # [0, +Inf] -> [from, +Inf]
		return ChiSquaredGenerator(4).next() + low
	elif low == -inf and high != inf: # [0, +Inf] -> [-Inf, to]
		return high - ChiSquaredGenerator(4).next()
	else:
		return UniformGenerator(low, high).next()

def getMonteCarloDensity(low, high, value):
	if low == -inf and high == inf:
		return normalDensity(0, 100, value)
	elif high == inf and low != -inf: # [0, +Inf] -> [from, +Inf]
		return chiSquaredDensity(4, value)
	elif low == -inf and high != inf: # [0, +Inf] -> [-Inf, to]
		return chiSquaredDensity(4, value)
	else:
		return uniformDensity(low, high, value)

class Limit:
	def __init__(self, low, high, is_function=False):
		self.low = low
		self.high = high
		self.is_function = is_function

class Integral:
	def __init__(self, func, limits):
		self.func = func
		self.limits = limits

def integrateMonteCarlo(integral, N):
	ksi = []
	for _ in range(N):
		values = []
		densities = []
		for limit in integral.limits:
			low = limit.low
			high = limit.high
			if limit.is_function:
				low = low(values[-1])
				high = high(values[-1])
			values.append(getMonteCarloValue(low, high))
			densities.append(getMonteCarloDensity(low, high, values[-1]))
		ksi.append(integral.func(*values) / prod(densities))
	return getExpectedValue(ksi)

def get_point(integral, expected):
	x = []
	y = []
	for i in range(1, 5):
		x.append(10 ** i)
		y.append(integrateMonteCarlo(integral, 10 ** i) - expected)
	return x, y

def show_plot(x, y, name, ax):
	ax.plot(x, y, c='b')
	ax.plot(x, [0 for i in range(len(x))], c='r')
	ax.set_title(name)

def main():
	_, axs = plt.subplots(2, 2)

	x, y = get_point(
		Integral(
			lambda x : exp(-1 * x ** 4), 
			[Limit(-inf, inf)]),
		1.8128)
	show_plot(x, y, "exp(-1 * x ** 4)", axs[0][0])

	x, y = get_point(
		Integral(
			lambda x, y: log(1 / sqrt(x ** 2 + y ** 2)), 
			[ Limit(-1, 1), Limit(lambda x: -1 * sqrt(1 - x ** 2), lambda x: sqrt(1 - x ** 2), True) ]),
		1.43315)
	show_plot(x, y, "log(1 / sqrt(x ** 2 + y ** 2))", axs[0][1])

	x, y = get_point(
		Integral(
			lambda x: tan(1 / x) / (x ** 2 + x - 3), 
			[Limit(2, inf)]),
		0.138364)
	show_plot(x, y, "tan(1 / x) / (x ** 2 + x - 3)", axs[1][0])

	x, y = get_point(
		Integral(
			lambda x, y: (x + 4) / (x ** 2 + y ** 4 + 1), 
			[Limit(-3, 4), Limit(-4, 3)]),
		28.2765)
	show_plot(x, y, "(x + 4) / (x ** 2 + y ** 4 + 1)", axs[1][1])

	plt.show()


main()