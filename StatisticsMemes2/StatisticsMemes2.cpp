#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <array>
#include <functional>

using namespace std;

class LinearCongruentialGenerator {
	unsigned long long a;
	unsigned long long betta;
	unsigned long long c;
	unsigned long long M;
public:
	LinearCongruentialGenerator(const long a, const long betta, const long c, const unsigned long M)
		: a(a), betta(betta), c(c), M(M) {}

	auto operator()() 
	{
		a = (a * betta + c) % M;
		return static_cast<double>(a) / M;
	}
};

class NegativeBinomialGenerator
{
	double c;
	int n;
	LinearCongruentialGenerator generator;
public:
	NegativeBinomialGenerator(const int n, const double p)
		: n(n), generator(79507, 79507, 0, 1ll << 31), c(1 / log(1 - p))
	{
	}

	auto operator()()
	{
		auto result = 0;
		for (auto i = 0; i < n; i++)
		{
			result += c * log(generator());
		}
		return result;
	}
};

class PoissonGenerator
{
	double mu;
	LinearCongruentialGenerator gen;
public:
	PoissonGenerator(const double mu)
		: mu(mu), gen(1, 65539, 0, 1ll << 31) {}

	auto operator()()
	{
		auto p = exp(-mu);
		auto x = 0l;
		auto r = gen() - p;
		while (r >= 0)
		{
			x++;
			p *= mu / x;
			r -= p;
		}
		return x;
	}
};

class BernoulliGenerator
{
	double p;
	LinearCongruentialGenerator gen;
public:
	BernoulliGenerator(const double p)
		: p(p), gen(79507, 79507, 0, 1ll << 31) {}
	
	auto operator()()
	{
		if (gen() < p)
		{
			return 1l;
		}
		else
		{
			return 0l;
		}
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

template<typename T, typename distributionFunction>
auto hiSquaredTest(const vector<T>& values, const distributionFunction& f)
{
	unordered_map<int, int> nu;
	for (const auto& value : values)
	{
		nu[value]++;
	}
	auto hiSquared = 0.0;
	for (const auto& value : nu)
	{
		const auto pk = f(value.first) * values.size();
		hiSquared += pow(value.second - pk, 2) / pk;
	}
	return hiSquared < approximateChiSquareQuantile(0.95, nu.size() - 1);
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
auto getSkewness(const vector<T>& data, const double mean)
{
	auto n = data.size();
	return sqrt(n * (n - 1)) / n / (n - 2) * accumulate(
		data.begin(),
		data.end(),
		0.0,
		[mean](const auto& acc, const auto& x) {
			return acc + pow(x - mean, 3);
	}) / pow(1.0 / n * accumulate(
		data.begin(),
		data.end(),
		0.0,
		[mean](const auto& acc, const auto& x) {
		return acc + pow(x - mean, 2);
	}), 1.5);
}

template<typename T>
auto getExcessKurtosis(const vector<T>& data, const double mean)
{
	auto n = data.size();
	return (pow(n, 2) - 1) / ((n - 2) * (n - 3)) * (1.0 / n * accumulate(
		data.begin(),
		data.end(),
		0.0,
		[mean](const auto& acc, const auto& x) {
			return acc + pow(x - mean, 4);
	}) / pow(1.0 / n * accumulate(
		data.begin(),
		data.end(),
		0.0,
		[mean](const auto& acc, const auto& x) {
			return acc + pow(x - mean, 2);
	}), 2) - 3 + 6.0 / (n + 1));
}

long C(const int n, const int k)
{
	if (k > n - k)
	{
		return C(n, n - k);
	}
	long res = 1;
	for (auto i = 0; i < k; i++) {
		res *= (n - k + i + 1) / (k - i);
	}
	return res;
}

constexpr long factorial(const int x)
{
	return x ? x * factorial(x - 1) : 1;
}

auto negativeBinomialPMF(const int n, const double p)
{
	return [n, p](const int x) {
		return C(x + n - 1, x) * pow(p, n) * pow(1 - p, x);
	};
}

auto puassonPMF(const double mu) 
{
	return [mu](const int x) {
		if (x < 0)
		{
			return 0.0;
		}
		return pow(mu, x) * exp(-mu) / factorial(x);
	};
}

auto bernouliPMF(const double p)
{
	return [p](const int x) {
		if (x == 0)
		{
			return 1 - p;
		}
		else
		{
			return p;
		}
	};
}

template<typename T, typename distributionFunction>
void showInfo(const T& generator, const double expectedMean, const double expectedDispersion, const double expectedSkewness, 
	const double expectedExcess, const distributionFunction& f)
{
	vector<int> values(1000);
	generate(values.begin(), values.end(), generator);
	const auto mean = getMean(values);
	cout << "Mean expected: " << expectedMean << " computed: " << mean << endl;
	cout << "Dispersion expected: " << expectedDispersion << " computed: " << getDispersion(values, mean) << endl;
	cout << "Skewness expected: " << expectedSkewness << " computed: " << getSkewness(values, mean) << endl;
	cout << "Excess kurtosis expected: " << expectedExcess << " computed: " << getExcessKurtosis(values, mean) << endl;
	cout << (hiSquaredTest(values, f) ? "passed chi-square test" : "failed chi-square test") << endl;
}

int main()
{
	cout << "Bernouli with 0.5" << endl;
	showInfo(BernoulliGenerator(0.5), 
		0.5, 
		0.5 * 0.5,
		(1 - 2 * 0.5) / sqrt((1 - 0.5) * 0.5),
		(1 - 6 * 0.5 * (1 - 0.5)) / (0.5 * (1 - 0.5)),
		bernouliPMF(0.5));
	cout << endl;

	cout << "Negative binomial with 5 0.25" << endl;
	showInfo(NegativeBinomialGenerator(5, 0.25), 
		5 * (1 - 0.25) / 0.25, 
		5 * (1 - 0.25) / pow( 0.25, 2 ),
		(1 + 0.25) / sqrt(0.25 * 5),
		6.0 / 5 + pow(1 - 0.25, 2) / (0.25 * 5),
		negativeBinomialPMF(5, 0.25));
	cout << endl;

	cout << "Bernouli with 0.6" << endl;
	showInfo(BernoulliGenerator(0.6),
		0.6,
		0.6 * (1 - 0.6),
		(1 - 2 * 0.6) / sqrt((1 - 0.6) * 0.6),
		(1 - 6 * 0.6 * (1 - 0.6)) / (0.6 * (1 - 0.6)),
		bernouliPMF(0.6));
	cout << endl;

	cout << "Poisson with 0.5" << endl;
	showInfo(PoissonGenerator(0.5), 
		0.5, 
		0.5, 
		1 / sqrt(0.5),
		1 / 0.5,
		puassonPMF(0.5));
	cout << endl;

	return 0;
}