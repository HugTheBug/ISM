// StatisticsMemes1.cpp : Defines the entry point for the application.
//

#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

class LinearCongruentialGenerator {
	long a;
	long betta;
	long c;
	unsigned long M;
public:
	LinearCongruentialGenerator(const long a, const long betta, const long c, const unsigned long M)
		: a( a ), betta( betta ), c( c ), M( M ) {}

	auto operator()() 
	{
		a = (a * betta + c) % M;
		return static_cast<double>(a) / M;
	}
};

class MultiplexialCongruentialGenerator {
	LinearCongruentialGenerator generator;
public:
	MultiplexialCongruentialGenerator(const long a, const long betta, const unsigned long M)
		: generator( a, betta, 0, M ) {}

	auto operator()() 
	{
		return generator();
	}
};

class MaclarenMarsagliaGenerator {
	LinearCongruentialGenerator xGen;
	LinearCongruentialGenerator yGen;
	unsigned long k;
	vector<double> v;
public:
	MaclarenMarsagliaGenerator(const LinearCongruentialGenerator& xGen, const LinearCongruentialGenerator& yGen, const unsigned long k)
		: xGen( xGen ), yGen( yGen ), k( k ), v( k )
	{
		generate(v.begin(), v.end(), xGen);
	}

	auto operator()() 
	{
		const auto x = xGen();
		const auto y = yGen();
		const auto j = k * y;

		const auto vJ = v[j];
		v[j] = x;

		return vJ;
	}
};

auto hiSquaredTest(const vector<double>& values, const unsigned long k)
{
	vector<long> nu(k);
	for (const auto& value : values)
	{
		nu[value * k]++;
	}
	const auto pk = static_cast<double>(values.size()) / k;
	auto hiSquared = 0.0;
	for (const auto& value : nu)
	{
		hiSquared += pow(value - pk, 2) / pk;
	}
	return hiSquared < 16.919;
}

auto kolmogorovTest(const vector<double>& values)
{
	auto copy = values;
	sort(copy.begin(), copy.end());
	auto dn = 0.0;
	auto i = 0.0;
	for (auto i = 0.0; i < copy.size(); i++)
	{
		auto d_max = max(copy[i] - i / copy.size(), (i + 1) / copy.size() - copy[i]);
		if (d_max > dn)
		{
			dn = d_max;
		}
	}
	dn *= sqrt(copy.size());
	return dn < 1.36;
}

int main()
{
	MultiplexialCongruentialGenerator generator(16807, 16807, 1ll << 31 );

	vector<double> g1(1000);
	generate(g1.begin(), g1.end(), generator);

	cout << "Multiplexial congruential generator:" << endl;
	cout << (hiSquaredTest(g1, 10) ? "passed hi-square" : "failed hi-square") << endl;
	cout << (kolmogorovTest(g1) ? "passed kolmogorov" : "failed kolmogorov") << endl;
	cout << endl;

	MaclarenMarsagliaGenerator another(
		LinearCongruentialGenerator(16807, 16807, 0, 1ll << 31),
		LinearCongruentialGenerator(16807 + 69, 16807 - 69, 69, 1ll << 31),
		64);

	vector<double> g2(1000);
	generate(g2.begin(), g2.end(), another);

	cout << "Maclaren Marsaglia generator:" << endl;
	cout << (hiSquaredTest(g2, 10) ? "passed hi-square" : "failed hi-square") << endl;
	cout << (kolmogorovTest(g2) ? "passed kolmogorov" : "failed kolmogorov") << endl;

	return 0;
}
