/**
  ICST347 Assignment #3, Part 1
  nBayes.cpp

  Purpose: 
  Classifies instances from test input as either male or female
  using Naive Bayes Classification

  Limitation:
  Specifically designed for the dataset collected in class

  @author Arazas, Al Jamil L. (2012-1-0712)
  @author Pascual, Louie Lester E. (2013-1-1743)
  @author Valeros, John Dinnex M. (2013-1-1618)

  @version 1.0 07/12/2016

  In our honor, we assure that we have not given nor received any
  unauthorized help in this work.

    Al Jamil L. Arazas, Louie Lester E. Pascual, John Dinnex M. Valeros

*/
#include <iostream>
#include <fstream>
#include <cmath>
#define PI 3.1415

using namespace std;

/**
  Computer the probability density function of a 
  standard normal distribution
   
  @param  value   given 
  @param  mean    computed mean
  @param  stdev   computed standard deviation
  @return density of a standard normal distribution?
*/
double dNorm(double value, double mean, double stdev) {
  double num = exp((-1 * pow(value-mean, 2.0)) / (2 * pow(stdev, 2.0)));
  double den = sqrt(2 * PI * pow(stdev,2.0));

  return num/den;
}


int main(int argc, char **argv) {
  // check command validity
  if (argc != 2) {
    cerr << "usage: " << argv[0] << " inputFile" << endl;
    return -1;
  }

  // input file
  ifstream fin;
  fin.open(argv[1]);

  // Setup STDEV and Mean
  double male_stdev_feet = 0.7398855017, 
          male_stdev_hair = 1.8103099796, 
          male_mean_feet = 9.4416666667, 
          male_mean_hair = 3.8666666667;
  double female_stdev_feet = 0.6946221995, 
        female_stdev_hair = 5.0283916138, 
        female_mean_feet = 8.65, 
        female_mean_hair = 11.9166666667;
  
  // Determines if the instance is a male or a female
  int run=1;
  int correct_male = 0; // correctly identified as male
  int correct_female = 0; // correctly identified as female
  int incorrect_male = 0; // incorrect identified as male
  int incorrect_female = 0; // incorrect identified as female

  int n;
  while(fin >> n) {
    
    string gender;
    double feet_length;
    double hair_length;
    
    fin >> gender >> feet_length >> hair_length;

    double nm, nf, evidence;

    nm = 0.5;
    nm *= dNorm(feet_length, male_mean_feet, male_stdev_feet);
    nm *= dNorm(hair_length, male_mean_hair, male_stdev_hair);

    nf = 0.5;
    nf *= dNorm(feet_length, female_mean_feet, female_stdev_feet);
    nf *= dNorm(hair_length, female_mean_hair, female_stdev_hair);

    evidence = nm + nf;

    double probm, probf;
    probm = nm/evidence;
    probf = nf/evidence;

    cout << "Run #" << run++ << endl;
    if(probm >= probf) {
      cout << "-------------------------\n";
      cout << "PREDICTION: \t" << n << " is a male " << endl;
      cout << "ACTUAL: \t" << n << " is a " << gender << endl;

      if(gender == "female") {
        cout << "RESULT: Incorrect\n";
        incorrect_male++;
      }
      else {
        cout << "RESULT: Correct\n";
        correct_male++;
      }
        
      cout << "-------------------------";
    }  
    else {
      cout << "-------------------------\n";
      cout << "PREDICTION: \t" << n << " is a female " << endl;
      cout << "ACTUAL: \t" << n << " is a " << gender << endl;

      if(gender == "male"){
        cout << "RESULT: Incorrect\n";
        incorrect_female++;
      }
      else {
        cout << "RESULT: Correct\n";
        correct_female++;
      }
        
      cout << "-------------------------";
    } 
    cout << endl << endl; 

  }

  cout << "REPORT for " << argv[1] << endl;
  cout << "-------------------------\n";
  cout << "Correctly predicted as male: \t\t" << correct_male << endl;
  cout << "Correctly predicted as female: \t\t" << correct_female << endl;
  cout << "Incorrectly predicted as male: \t\t" << incorrect_male << endl;
  cout << "Incorrectly predicted as female: \t" << incorrect_female << endl;
  cout << "-------------------------\n\n";



}

