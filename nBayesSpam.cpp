/**
  ICST347 Assignment #3
  nBayesSpam.cpp

  Purpose: 
  Classifies instances from test input either as spam or non-spam
  using Naive Bayes Classification

  Limitation:
  Specifically designed for the dataset collected in class

  @author Arazas, Al Jamil L. (2012-1-0712)
  @author Pascual, Louie Lester E. (2013-1-1743)
  @author Valeros, John Dinnex M. (2013-1-1618)

  @version 1.0 07/21/2016

  In our honor, we assure that we have not given nor received any
  unauthorized help in this work.

    Al Jamil L. Arazas, Louie Lester E. Pascual, John Dinnex M. Valeros

*/
#include <iostream>
#include <cmath>
#include <fstream>
using namespace std;

#define PI 3.1415
#define MAXCOL 58

/**
  Computes the probability density function of a 
  standard normal distribution
   
  @param  value   given 
  @param  mean    computed mean
  @param  stdev   computed standard deviation
  @return density of a standard normal distribution?
*/
double dNorm(double value, double mean, double stdev) {
	if(stdev == 0)
		return 0;

  	double num = exp((-1 * pow(value-mean, 2.0)) / (2 * pow(stdev, 2.0)));
  	double den = sqrt(2 * PI * pow(stdev,2.0));

  	return num/den;
}

int main()
{
	// Standard Deviation and Mean Calculations
	ifstream fin;
	fin.open("training.in");

	double data[MAXCOL];
	double sum[MAXCOL] = {0.0};
	double spamsum[MAXCOL] = {0.0};
	double notspamsum[MAXCOL] = {0.0};
	double sum2[MAXCOL] = {0.0};
	double spamsum2[MAXCOL] = {0.0};
	double notspamsum2[MAXCOL] = {0.0};
	double sd[MAXCOL] = {0.0};
	double spamsd[MAXCOL] = {0.0};
	double notspamsd[MAXCOL] = {0.0};
	int ctr = 0;
	int spamctr = 0;
	int notspamctr = 0;
	double p;
	char c;
	while (fin>>p)
	{
		ctr++;
		sum[0] += p;
		sum2[0] += (p*p);
		data[0]=p;
		//cout << p << " ";
		for (int i=1;i<MAXCOL;i++)
		{
			fin>>c>>data[i];
			sum[i] += data[i];
			sum2[i] += data[i]*data[i];
			//cout << data[i] << " ";
		}

		if (data[MAXCOL-1]==0)
		{
			notspamctr++;
			for (int i=0;i<MAXCOL;i++)
			{
				notspamsum[i] += data[i];
				notspamsum2[i] += data[i]*data[i];
			}
		}
		else
		{
			spamctr++;
			for (int i=0;i<MAXCOL;i++)
			{
				spamsum[i] += data[i];
				spamsum2[i] += data[i]*data[i];
			}
		}
		//cout<<endl;

	}


	for (int i=0;i<MAXCOL;i++)
	{
		
		double c = ctr;
		sd[i] = sqrt((c / (c - 1))*((sum2[i]/c)-pow(sum[i]/c,2)));
		//cout<<sd[i]<<endl;

		c = spamctr;
		spamsd[i] = sqrt((c / (c - 1))*((spamsum2[i]/c)-pow(spamsum[i]/c,2)));
		//cout<<spamsd[i]<<endl;

		c = notspamctr;
		notspamsd[i] = sqrt((c / (c - 1))*((notspamsum2[i]/c)-pow(notspamsum[i]/c,2)));
		cout<<notspamsd[i]<<endl;
		
		//note
		//mean would be sum[i]/ctr

	}

	fin.close();


	/** **/
	fin.open("test.in");

	// Determines if the instance is spam or not spam
  	int run=0;
  	int correct_spam = 0; // correctly identified as spam
  	int correct_nspam = 0; // correctly identified as not spam
  	int incorrect_spam = 0; // incorrect identified as spam
  	int incorrect_nspam = 0; // incorrect identified as not spam

  	double n;
  	double inputData[MAXCOL];
  	while(fin >> n) {
	   	run++;

	    // get test data
	   	inputData[0] = n;
	   	char c;

	    for(int i=1; i<MAXCOL; i++) {
	    	fin >> c >> inputData[i];
	    }
	   
	    double n_spam, n_nspam, evidence;

	    // computes n_spam
	    n_spam = 0.5;
	    for(int i=0; i<MAXCOL-1; i++) {
	    	double temp;
	    	temp = dNorm(inputData[i], spamsum[i]/spamctr, spamsd[i]);

	    	if(temp != 0)
	    		n_spam *= dNorm(inputData[i], spamsum[i]/spamctr, spamsd[i]);

	    }

	    n_nspam = 0.5;
	    for(int i=0; i<MAXCOL-1; i++) {
	    	double temp;
	    	temp = dNorm(inputData[i], notspamsum[i]/notspamctr, notspamsd[i]);

	    	if(temp != 0)
	    		n_nspam *= dNorm(inputData[i], notspamsum[i]/notspamctr, notspamsd[i]);
	    }
	    
	    evidence = n_spam + n_nspam;

	    double probspam, probnspam;
	    probspam = n_spam/evidence;
	    probnspam = n_nspam/evidence;

	    //cout << probspam << " " << probnspam << endl;

	    
	    
	    if(probspam >= probnspam) {
	      cout << "-------------------------\n";
	      cout << "PREDICTION: \t" << run << " is a spam " << endl;

	      if(inputData[MAXCOL-1] == 1) {
	      	cout << "ACTUAL: \t" << run << " is a spam" << endl;
	      }
	      else {
	      	cout << "ACTUAL: \t" << run << " is not a spam" << endl;	
	      }
	      

	      if(inputData[MAXCOL-1] == 0) {
	        cout << "RESULT: Incorrect\n";
	        incorrect_spam++;
	      }
	      else {
	        cout << "RESULT: Correct\n";
	        correct_spam++;
	      }
	        
	      cout << "-------------------------";
	    }  
	    else {
	      cout << "-------------------------\n";
	      cout << "PREDICTION: \t" << run << " is not a spam " << endl;

	      if(inputData[MAXCOL-1] == 1) {
	      	cout << "ACTUAL: \t" << run << " is a spam" << endl;
	      }
	      else {
	      	cout << "ACTUAL: \t" << run << " is not a spam" << endl;	
	      }

	      if(inputData[MAXCOL-1] == 1){
	        cout << "RESULT: Incorrect\n";
	        incorrect_nspam++;
	      }
	      else {
	        cout << "RESULT: Correct\n";
	        correct_nspam++;
	      }
	        
	      cout << "-------------------------";
	    } 

	    cout << endl << endl; 
  	}


	cout << "-------------------------\n";
	cout << "Correctly predicted as spam: \t\t" << correct_spam << endl;
	cout << "Correctly predicted as not spam: \t\t" << correct_nspam << endl;
	cout << "Incorrectly predicted as spam: \t\t" << incorrect_spam << endl;
	cout << "Incorrectly predicted as not spam: \t" << incorrect_nspam << endl;
	cout << "-------------------------\n\n";


	return 0;
}