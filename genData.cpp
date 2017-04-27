/**
  ICST347 Assignment #3
  genData.cpp

  Purpose: 
  This program serves to distribute a set of data to
  two input files, namely training.in and test.in.

  @author Arazas, Al Jamil L. (2012-1-0712)
  @author Pascual, Louie Lester E. (2013-1-1743)
  @author Valeros, John Dinnex M. (2013-1-1618)

  @version 1.0 07/21/2016

  In our honor, we assure that we have not given nor received any
  unauthorized help in this work.

    Al Jamil L. Arazas, Louie Lester E. Pascual, John Dinnex M. Valeros

*/
    
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
using namespace std;

#define MAXCOL 58

int main()
{
	ifstream fin("spambase.data");
	ofstream foutTr("training.in");
	ofstream foutTe("test.in");	
	string data;
	while (fin>>data)
	{
		string t;
		getline(fin,t);
		data += t;
		double v = rand() / double(RAND_MAX);
		if ( v < 0.6 ) 
		{
			foutTr << data<<endl;
		} 
		else 
		{
			foutTe << data<<endl;
		}
	}


	return 0;
}