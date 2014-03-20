#include <math.h>
#include <fstream>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <sstream>
#include <iostream>
using namespace std;

// vector< vector<double> > readDistributionCandidate(string filename)
// {
// 	vector< vector<double> > result;
// 	string line;
// 	ifstream infile(filename);
// 	double temp;
// 	while(getline(infile, line))
// 	{
// 		vector<double> v;
// 		istringstream iss(line);
// 		while(iss >> temp)
// 		{
// 			v.push_back(temp);
// 		}
// 		result.push_back(v);
// 	}
// 	return result;
// }

vector<double> readDistributionDocument (string filename)
{
	vector<double> result;
	string line;
	ifstream infile(filename);
	double temp;
	while(getline(infile, line))
	{
		istringstream iss(line);
		while(iss >> temp)
		{
			result.push_back(temp);
		}
	}
	return result;
}

double kl_divergence(vector<double> p, vector<int > q)
{
	double sum = 0.0;
	for(int i = 0; i < (int)p.size(); i++)
	{
		sum += p[i] * (log10(p[i]) - log10(q[i]));
		printf("%.2f %.2f %.2f\n", p[i], q[i], sum);
	}
	return sum;

}

int main (int argc, char** argv[])
{
	int n_topics = (int)argv[1];
	int n_candidates = (int)argv[2];
	double sum;
    vector<double> p = readDistributionDocument("p.txt");
	vector<double> q = readDistributionDocument("q.txt");
	sum = kl_divergence(p, q);
	return 0;
}

