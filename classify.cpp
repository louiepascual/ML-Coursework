/**
  ICST347 Assignment #5, Part 1
  classify.cpp

  Purpose: 
  Produces a machine learning model of the training set of tennisData.txt
  and classifies the data in the test set using Naive Bayes Classification.

  Limitation:
  Specifically designed for the dataset collected in class

  @author Arazas, Al Jamil L. (2012-1-0712)
  @author Pascual, Louie Lester E. (2013-1-1743)
  @author Valeros, John Dinnex M. (2013-1-1618)

  @version 1.0 07/26/2016

  In our honor, we assure that we have not given nor received any
  unauthorized help in this work.

    Al Jamil L. Arazas, Louie Lester E. Pascual, John Dinnex M. Valeros

*/
#include <iostream>
#include <fstream>
#include <cmath>
#include <map>
#include <string>
#include <vector>
using namespace std;

// Data Declarations
map< string,map<string,double> > P;

int main(int argc, char **argv) {
	// Check command validity
	if(argc != 3) {
		cerr << "usage: " << argv[0] << " trainingFile testFile" << endl;
    	return -1;
	}

	// Get training and test file
	string trainingFile, testFile;
	trainingFile = argv[1];
	testFile = argv[2];

	/**
	Training
	**/
	ifstream fin;
	fin.open(trainingFile.c_str());
	
	int n = 0;
	fin>>n;

	int q = 0;
	fin>>q;

	vector <string> labels;
	for (int i = 0;i<q;i++)
	{
		string t;
		fin>>t;
		labels.push_back(t);
		if (labels[i] != "no.tennis" || labels[i] != "play.tennis")
		{
		P["play.tennis"][labels[i]] = 0;
		P["no.tennis"][labels[i]] = 0;
		}
	}

	P["play.tennis"][""] = 0;
	P["no.tennis"][""] = 0;
	
	for (int i = 0; i < n; i++)
	{
		int r = 0;
		fin>>r;

		//string temp_arr[r];
		vector <string> temp_arr;

		for (int j=0;j<r;j++)
		{
			string t;
			fin>>t;
			temp_arr.push_back(t);
		}
		for (int j=0;j<r-1;j++)
		{
			P[temp_arr[r-1]][temp_arr[j]]++;
		}
		P[temp_arr[r-1]][""]++;
	}

	for (int i = 0; i < q ; i++)
	{
		P["play.tennis"][labels[i]] /= P["play.tennis"][""];
		P["no.tennis"][labels[i]] /= P["no.tennis"][""];
	}

	P["play.tennis"][""] = P["play.tennis"][""] / n;
	P["no.tennis"][""] = P["no.tennis"][""] / n;

	fin.close();

	/**
	Testing
	**/

	// Open testFile
	fin.open(testFile.c_str());

	// Get number of tests
	fin >> n;

	// Disregard Attribute-Value line
	int t;
	string trash;
	fin >> t;
	for(int i=0; i<t; i++) {fin >> trash;};

	string outlook, temp, humidity, wind, actualResult;
	int truePositive = 0;
	int trueNegative = 0; 
	int falsePositive = 0;
	int falseNegative = 0;

	for(int i=0; i<n; i++) {
		fin >> t >> outlook >> temp >> humidity >> wind >> actualResult;

		cout << "Run # " << i+1 << endl;
		// Computes the test input probability to be a yes.tennis and a no.tennis
		double probyes, probno;
		probyes = (P["play.tennis"][""] * P["play.tennis"][outlook] * P["play.tennis"][temp] * P["play.tennis"][humidity] * P["play.tennis"][wind]);
		probno = (P["no.tennis"][""] * P["no.tennis"][outlook] * P["no.tennis"][temp] * P["no.tennis"][humidity] * P["no.tennis"][wind]);

		// Makes a decision
		if(probyes >= probno) {
			// This means that naiveBayes predicted it is as yes.tennis
			cout << "Prediction: play.tennis" << endl;
			cout << "Actual: " << actualResult << endl;

			if(actualResult == "play.tennis") {
				truePositive++;
				cout << "Result: Correct!" << endl;
			}
			else {
				falsePositive++;
				cout << "Result: Incorrect!" << endl;
			}
		}
		else {
			// This means that naiveBayes predicted it is as no.tennis
			cout << "Prediction: no.tennis" << endl;
			cout << "Actual: " << actualResult << endl;

			if(actualResult == "no.tennis") {
				trueNegative++;
				cout << "Result: Correct!" << endl;
			}
			else {
				falseNegative++;
				cout << "Result: Incorrect!" << endl;
			}
		}
		cout << endl;
	}

	fin.close();

	/**
	Results
	**/

	cout << "Number correctly predicted as no.tennis:\t " << trueNegative << endl;
	cout << "Number correctly predicted as play.tennis:\t " << truePositive << endl;
	cout << "Number incorrectly predicted as no.tennis:\t " << falseNegative << endl;
	cout << "Number incorrectly predicted as play.tennis:\t " << falsePositive << endl;

}
