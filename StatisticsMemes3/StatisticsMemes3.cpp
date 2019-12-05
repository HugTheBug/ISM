#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>
#include <numeric>

using namespace std;

const auto pi = 3.1415926535897932384626433;

class LinearCongruentialGenerator {
	unsigned long long a;
	unsigned long long betta;
	unsigned long long c;
	unsigned long long M;
public:
	LinearCongruentialGenerator(const long a = 16387, const long betta = 16387, const long c = 0, const unsigned long long M = 1ll << 31)
		: a(a), betta(betta), c(c), M(M) {}

	auto operator()()
	{
		a = (a * betta + c) % M;
		return static_cast<double>(a) / M;
	}
};

class LinearBased {
protected:
	LinearCongruentialGenerator generator;
};

class GeneratorGenerator : public LinearBased {
	LinearCongruentialGenerator generator;
	unsigned long long M = 1ll << 31;
public:
	auto operator()()
	{
		return LinearCongruentialGenerator(
			M * generator(),
			M * generator(),
			0,
			M
		);
	}
};

class ArrayBased {
protected:
	vector<LinearCongruentialGenerator> generators;

public:
	ArrayBased( const int n = 64 )
	{
		generate_n(back_inserter(generators),
			n, GeneratorGenerator());
	}
};

class NormalGenerator : public ArrayBased {
	const double s;
	const double m;
	int n;

public:
	NormalGenerator(const double m, const double sSquare, const int n = 64)
		: m(m), s(sqrt(sSquare)), ArrayBased(n), n(n)
	{}

	auto operator()()
	{
		auto sum = 0.0;
		for (auto& generator : generators)
		{
			sum += generator();
		}
		return m + s * sqrt(12.0 / n) * (sum - n / 2.0);
	}
};

class LaplaceGenerator : public LinearBased
{
	double lambda;
	double lastRandom;
public:
	LaplaceGenerator(const double lambda)
		: lambda(lambda)
	{
		lastRandom = generator();
	}

	auto operator()()
	{
		const auto newRandom = generator();
		const auto result = 1 / lambda * log(lastRandom / newRandom);
		lastRandom = newRandom;
		return result;
	}
};

class ExponentialGenerator : public LinearBased
{
	const double lambda;
public:
	ExponentialGenerator( const double lambda)
		: lambda(lambda)
	{}

	auto operator()()
	{
		return -1 / lambda * log(generator());
	}
};

class WeibullGenerator : public LinearBased
{
	const double a;
	const double c;
public:
	WeibullGenerator(const double a, const double c)
		: a(a), c(c)
	{}

	auto operator()()
	{
		return a * pow(-log(generator()), 1 / c);
	}
};

class LogisticGenerator : public LinearBased
{
	double a;
	double b;
public:
	LogisticGenerator(const double a, const double b)
		: a(a), b(b)
	{
	}

	auto operator()()
	{
		const auto r = generator();
		return a - b * log((1 - r) / r);
	}
};

class NormalDistribution
{
	double m;
	double sSquare;
public:
	NormalDistribution(const double m, const double sSquare)
		: m(m), sSquare(sSquare)
	{}

	auto getMean() const
	{
		return m;
	}

	auto getDispersion() const
	{
		return sSquare;
	}

	auto getCDF() const
	{
		return [this](const auto& x)
		{
			return (1 + erf((x - this->m) / sqrt(2 * this->sSquare))) / 2;
		};
	}

	auto getName() const
	{
		return "Normal";
	}
};

class ExponentialDistribution
{
	double lambda;
public:
	ExponentialDistribution(const double lambda)
		: lambda(lambda)
	{}
	auto getMean() const
	{
		return 1.0 / lambda;
	}

	auto getDispersion() const
	{
		return 1.0 / lambda / lambda;
	}

	auto getCDF() const
	{
		return [this](const auto& x)
		{
			return 1 - exp(-this->lambda * x);
		};
	}

	auto getName() const
	{
		return "Exponential";
	}
};

class WeibullDistribution
{
	double a;
	double c;
public:
	WeibullDistribution(const double a, const double c)
		: a(a), c(c)
	{}

	auto getMean() const
	{
		return a * tgamma(1 + 1.0 / c);
	}

	auto getDispersion() const
	{
		return pow(a, 2) * (tgamma(1 + 2.0 / c) - pow(tgamma(1 + 1.0 / c), 2));
	}

	auto getCDF() const
	{
		return [this](const auto& x)
		{
			return x >= 0 ? 1 - exp(-pow(x / this->a, this->c)) : 0;
		};
	}

	auto getName() const
	{
		return "Weibull";
	}
};

class LogisticDistribution
{
	double a;
	double b;
public:
	LogisticDistribution(const double a, const double b)
		: a(a), b(b)
	{}

	auto getMean() const
	{
		return a;
	}

	auto getDispersion() const
	{
		return pow(b * pi, 2) / 3;
	}

	auto getCDF() const
	{
		return [this](const auto& x)
		{
			return 1.0 / (1 + exp(-1 * (x - this->a) / this->b));
		};
	}

	auto getName() const
	{
		return "Logistic";
	}
};

class LaplaceDistribution
{
	double lambda;
public:
	LaplaceDistribution(const double lambda)
		:lambda(lambda)
	{}

	auto getMean() const
	{
		return 0.0;
	}

	auto getDispersion() const
	{
		return 2.0 / lambda / lambda;
	}

	auto getCDF() const
	{
		return [this](const auto& x)
		{
			return x < 0 ? 0.5 * exp(this->lambda * x) : 1 - 0.5 * exp(this->lambda * -x);
		};
	}

	auto getName() const
	{
		return "Laplace";
	}
};

auto approximateChiSquareQuantile(const double alpha, const int n)
{
	const static array<double, 7> a = { 1.0000886, 0.4713941, 0.000134802,
		-0.008553069, 0.00312558, -0.0008426812, 0.00009780499 };

	const static array<double, 7> b = { -0.2237368, 0.02607083, 0.01128186,
		-0.01153761, 0.005169654, 0.00253001, 0.001450117 };

	const static array<double, 7> c = { -0.01513904, -0.008986007, 0.02277679,
		-0.01323293, -0.006950356, 0.001060438, 0.001565326 };

	auto result = 0.0;
	const auto d = alpha < 0.5 ?
		-2.0637 * pow((log(1 / alpha) - 0.16), 0.4274) + 1.5774 :
		2.0637 * pow((log(1 / (1 - alpha)) - 0.16), 0.4274) - 1.5774;

	for (auto i = 0; i < a.size(); i++)
	{
		result += pow(n, static_cast<double>(i) / -2)* pow(d, i)* (a[i] + b[i] / n + c[i] / pow(n, 2));
	}

	return pow(result, 3) * n;
}

template<typename T>
auto getDispersion(const vector<T>& data, const double mean)
{
	return accumulate(
		data.begin(),
		data.end(),
		0.0,
		[mean](const auto& acc, const auto& x) {
		return acc + pow(x - mean, 2);
	}) / static_cast<double>(data.size());
}

template<typename T>
auto getMean(const vector<T>& data)
{
	return accumulate(
		data.begin(),
		data.end(),
		0.0,
		std::plus<double>()) / static_cast<double>(data.size());
}

template<typename T, typename distributionFunction>
auto hiSquaredTest(const vector<T>& values, const distributionFunction& f)
{
	auto copy = values;
	sort(copy.begin(), copy.end());
	const auto length = 10.0;
	auto hiSquared = 0.0;
	for (int i = 0; i < length - 1; i++)
	{
		const auto pk = (f(copy[copy.size() * (i + 1) / length]) - f(copy[copy.size() * i / length])) * copy.size();
		hiSquared += pow(copy.size() / length - pk, 2) / pk;		
	}
	return hiSquared < approximateChiSquareQuantile(0.95, length - 1);
}

template<typename T, typename distributionFunction>
auto kolmogorovTest(const vector<T>& values, const distributionFunction& f)
{
	auto copy = values;
	sort(copy.begin(), copy.end());

	auto dn = 0.0;
	for (auto i = 0.0; i < copy.size(); i++)
	{
		dn = max(dn, abs(i / copy.size() - f(copy[i])));
	}
	return dn * sqrt(copy.size()) < 1.36;
}

template<typename DistT, typename GenT>
auto showInfo(const DistT& distribution, const GenT& generator)
{
	vector<double> values(1000);
	generate(values.begin(), values.end(), generator);
	const auto mean = getMean(values);
	cout << distribution.getName() << endl;
	cout << "Mean expected: " << distribution.getMean() << " computed: " << mean << endl;
	cout << "Dispersion expected: " << distribution.getDispersion() << " computed: " << getDispersion(values, mean) << endl;
	cout << (hiSquaredTest(values, distribution.getCDF()) ? "passed chi-square test" : "failed chi-square test") << endl;
	cout << (kolmogorovTest(values, distribution.getCDF()) ? "passed kolmogorov test" : "failed kolmogorov test") << endl;
	cout << endl;
}

int main()
{
	showInfo(NormalDistribution(4, 25), NormalGenerator(4, 25));
	showInfo(NormalDistribution(0, 1), NormalGenerator(0, 1));
	showInfo(ExponentialDistribution(0.5), ExponentialGenerator(0.5));
	showInfo(WeibullDistribution(4, 0.5), WeibullGenerator(4, 0.5));
	showInfo(LogisticDistribution(2, 3), LogisticGenerator(2, 3));
	showInfo(LaplaceDistribution(2), LaplaceGenerator(2));
}