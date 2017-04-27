/**
  ICST347 Assignment #1, item 1 
  id3.cpp

  Purpose: 
  This program creates the machine learning model/rules
  using ID3 algorithm.

  @author Arazas, Al Jamil L. (2012-1-0712)
  @author Pascual, Louie Lester E. (2013-1-1743)
  @author Valeros, John Dinnex M. (...)

  @version 1.0 06/21/2016

  In our honor, we assure that we have not given nor received any
  unauthorized help in this work.

    Al Jamil L. Arazas, Louie Lester E. Pascual, John Dinnex M. Valeros

*/

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <fstream>

using namespace std;

// data
vector <string> name;
vector < vector <string> > S;

map <int, vector<string> > av;

void init()
{
	av[0].push_back("1st");
	av[0].push_back("2nd");
	av[0].push_back("3rd");
	av[0].push_back("crew");

	av[1].push_back("female");
	av[1].push_back("male");

	av[2].push_back("adult");
	av[2].push_back("child");
}


int count (vector <int> Ex, int TAttr, string value) 
{
	int numCount = 0; // counter
	for (int i = 0; i < Ex.size(); i++) 
	{
		int r = Ex[i];
		if (S[r][TAttr] == value)
			numCount++;
	}
	return numCount;
}

int count (vector <int> Ex, int A, string v, int TAttr, string value) 
{
	int numCount = 0;
	for(int i = 0; i < Ex.size(); i++)
	{
		int r = Ex[i];
		if (S[r][A] == v && S[r][TAttr] == value)
			numCount++; 
	}
	return numCount;
}

double H(vector <int> Ex, int TAttr) 
{
	int pos = count(Ex, TAttr, "yes");
	int neg = count(Ex, TAttr, "no");

	if ( pos == 0) return 0.0;
	if ( neg == 0) return 0.0;

	double total = pos + neg;
	double ppos = pos/total;
	double pneg = neg/total;

	return -ppos * log2(ppos) - pneg * log2(pneg);
}

double H(vector <int> Ex, int A, string v, int TAttr)
{
	int pos = count(Ex, A, v, TAttr, "yes");
	int neg = count(Ex, A, v, TAttr, "no");

	if ( pos == 0) return 0.0;
	if ( neg == 0) return 0.0;

	double total = pos + neg;
	double ppos = pos/total;
	double pneg = neg/total;

	return -ppos * log2(ppos) - pneg * log2(pneg);
}

double avgH(vector <int> Ex, int A, int TAttr)
{

	double sumH = 0.0;
	for(int i = 0; i < av[A].size(); i++)
		sumH += H(Ex, A, av[A][i], TAttr);
	return (sumH / av[A].size());

}

int bestAttr (vector <int> Ex, vector <int> Attr, int TAttr)
{

	int bestA = Attr[0];
	double bestH = avgH(Ex, Attr[0], TAttr);

	for (int i = 1; i < Attr.size(); i++) 
	{
		double tH = avgH(Ex, Attr[i], TAttr);
		if (tH < bestH){
			bestH = tH;
			bestA = Attr[i];
		}
	}

	return bestA;

}

struct Node 
{
	string label;
	string value;
	vector <Node *> child;

};

Node * ID3(vector <int> Ex, vector <int> Attr, int TAttr) 
{
	Node *root = new Node();
	int pos, neg;


	root = new Node();
	pos = count(Ex, TAttr, "yes");
	neg = count(Ex, TAttr, "no");

	if (neg == 0)
	{
		root->label = "yes";
		return root;
	}


	if (pos == 0)
	{
		root->label = "no";
		return root;
	}
	if(Attr.size() == 0) {
		root->label = ((pos > neg) ? "yes" : "no");
		return root;

	}
	else 
	{
		int A;
		map <string, vector <int> > newEx;

		A = bestAttr(Ex, Attr, TAttr);
		root->label = name[A];
		for(int j = 0; j < Ex.size(); j++)
			{
				int r = Ex[j];
				newEx[S[r][A]].push_back(r);
			}

		for (int i = 0; i < av[A].size(); i++) 
		{
			string v = av[A][i];
			
			if (newEx[v].size() == 0)
			{
				Node *n = new Node();
				int pos = count(newEx[v], A, v, TAttr, "yes");
				int neg = count(newEx[v], A, v, TAttr, "no");
				n->label = (( pos > neg) ? "yes" : "no");
				n->value = v;
				root->child.push_back(n);
			}
			else
			{
				vector <int> newAttr;
				for(int i = 0; i < Attr.size(); i++)
				{
					if ( Attr[i] != A)
						newAttr.push_back(Attr[i]);
				}
				Node *n = ID3(newEx[v], newAttr , TAttr);
				n->value = v;
				root->child.push_back(n);

			}
		}
		return root;
	}
	return NULL;
}

void displayTree(Node *t, int level) {
	if ( t != NULL) {
		for(int i = 0; i < level; i++)
			cout << "	|";
		if (t->value == "")
			cout << "IF " << t->label << endl;
		else if (t->child.size() == 0)
			cout << " = " <<  t->value << " THEN Survived = " << t->label << endl;
		else
			cout << " = " <<  t->value << " AND IF " << t->label << endl;
		for(int i = 0; i < t->child.size(); i++)
			displayTree(t->child[i], level+1);
	}
}

int main(int argc, char **argv)
{
	if (argc != 2) 
	{
		cerr << "usage: " << argv[0] << " inputFile" << endl;
		return -1;
	}
	
	ifstream fin(argv[1]); 
	int r,c;
	fin	>>	r  >>	c;
	
	for (int i=0;i<c;i++)
	{
		string t;
		fin>>t;
		name.push_back(t);
	}

	for (int j=0;j<r;j++)
	{
		vector <string> tvc;
		for (int i=0;i<c;i++)
		{
			string data;
			fin>>data;
			tvc.push_back(data);
		}
		S.push_back(tvc);
	}

	init();
	
	vector <int> Ex;
	vector <int> Attr;
	int TAttr;
	
	for( int i = 0; i < r; i++) Ex.push_back(i);
	for( int i = 0; i < c-1; i++)	Attr.push_back(i);

	TAttr = c-1;
	
	Node *dTree = ID3(Ex, Attr, TAttr);
	displayTree(dTree,0);

	fin.close(); // close the file
	return 0;

}
