/**
  ICST347 Assignment #2, item 1
  genRules.cpp

  Purpose: 
  Generates the rules and outputs the lift, confidence, and support

  @author Arazas, Al Jamil L. (2012-1-0712)
  @author Pascual, Louie Lester E. (2013-1-1743)
  @author Valeros, John Dinnex M. (2013-1-1618)

  @version 1.0 07/04/2016

  In our honor, we assure that we have not given nor received any
  unauthorized help in this work.

    Al Jamil L. Arazas, Louie Lester E. Pascual, John Dinnex M. Valeros

*/

#include "global.h"

vector<string> name;
map <string,int> nameMap;
vector< vector<bool> > D;
int ROWS, COLS;
struct Rules {
  itemSet lhs;
  itemSet rhs;

  double support;
  double confidence;
  double lift;

  bool operator > (const Rules& str) const {
    if(lift > str.lift)
      return true;
    else if(confidence > str.confidence)
      return true;
    else if(support > str.support)
      return true;
    return false;
}
  
};

vector<Rules> ruleSet;
void ruleGen(int sizeLimit, itemSet item);
void outputRule(Rules r);
void pairSets(candSet l, candSet r);
bool contains(itemSet x, itemSet y);
void outputItemSet(itemSet x);
itemSet unionSet(itemSet x, itemSet y);
double support(Rules r);
double confidence(Rules r);
double lift(Rules r);
void computeMeasures(Rules &r);


void init(string fileName) {
  // Start processing of input data
  ifstream fin(fileName);

  // Get Number of Transactions
  fin >> ROWS;

  // Get Number of Items
  fin >> COLS;

  // Setup NameSet
  for(int i=0; i<COLS; i++) {
    string itemName;
    fin >> itemName;
    name.push_back(itemName);
    nameMap[itemName] = i;
  }

  // Setup DataSet
  for(int i=0; i<ROWS; i++) {
    int numItems;
    string itemName;

    // Setup a transaction
    fin >> numItems;
    vector<bool> transaction(COLS, 0);
    for(int j=0; j<numItems; j++) {
      fin >> itemName;
      transaction[nameMap[itemName]] = 1;
    }

    // Push a transaction to dataset
    D.push_back(transaction);
  } 
}



int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "usage: " << argv[0] << " inputFile minSupport" << endl;
    return -1;
  }
  // Initialize dataset
  init(argv[1]);

  // User Apriori to get freqSet
  int k = stoi(argv[2]);
  freqSet L = apriori(k);
  //display(L);


  // GENERATE RULES
  for(int i=0; i<L.size(); i++) {
    freq temp = L[i];
    itemSet item = temp.I;
    int size = temp.I.size();

    if(size <= 1) { // Disregard frequencies with only 1
      continue;
    }
    else if(size == 2) {
      Rules r1, r2;

      r1.lhs.push_back(item[0]);
      r1.rhs.push_back(item[1]);
      ruleSet.push_back(r1);

      r2.lhs.push_back(item[1]);
      r2.rhs.push_back(item[0]);
      ruleSet.push_back(r2);
    }
    else {
      int sizeLimit = size-2;
      for(int i=1; i<=sizeLimit; i++) {
    
        candSet l = genSubset(i, size, item);
        candSet r = genSubset(size-i, size, item);

        pairSets(l,r);
      } 
    }
    
  }
  // Computer First
  for(int i=0; i<ruleSet.size(); i++) {
    computeMeasures(ruleSet[i]);
  }

  // Then sort
  sort(ruleSet.begin(), ruleSet.end(), greater<Rules>());

  // Then output
  for(int i=0; i<ruleSet.size(); i++) {
    outputRule(ruleSet[i]);
    cout << endl;
  }

 

  return 0;
}    


void outputRule(Rules r) {
  cout << "{";
  for(int i=0; i<r.lhs.size()-1; i++) {
    cout << name[r.lhs[i]] << ", ";
  }
  cout << name[r.lhs[r.lhs.size()-1]] << "} => {";

  for(int i=0; i<r.rhs.size()-1; i++) {
    cout << name[r.rhs[i]] << ", ";
  }
  cout << name[r.rhs[r.rhs.size()-1]] << "}";
  
  cout << "\t lift: " << r.lift;
  cout << "\t cnfd: " << r.support;
  cout << "\t supp: " << r.confidence;
  
}

void outputItemSet(itemSet x) {
  cout << "{";
  for(int i=0; i<x.size()-1; i++)
    cout << x[i] << ", ";
  cout << x[x.size()-1] << "}\n";
}

void pairSets(candSet l, candSet r) {
  for(int i=0; i<l.size(); i++) {
    for(int j=0; j<r.size(); j++) {
      if(match(l[i],r[j]))
        continue;
      else if(contains(l[i], r[j])) // does itemset r[j] contains itemset l[i]
        continue;
      else {
        Rules r1, r2;
        r1.lhs = l[i];
        r1.rhs = r[j];
        ruleSet.push_back(r1);

        r2.lhs = r[j];
        r2.rhs = l[i];
        ruleSet.push_back(r2);
      }
    }
  }
}

bool contains(itemSet x, itemSet y) {
  if(x.size() > y.size())
    return false;
  else {
    for(int i=0; i<x.size(); i++) {
      for(int j=0; j<y.size(); j++) {
        if(x[i] == y[j])
          return true;
      }
    }
  }
  return false;
}

itemSet unionSet(itemSet x, itemSet y) {
  itemSet t;
  for(int i=0; i<x.size(); i++) {
    t.push_back(x[i]);
  }
  for(int i=0; i<y.size(); i++) {
    if(!(find(t.begin(), t.end(), y[i]) != t.end())) {
      t.push_back(y[i]);
    }
  }
  return t;
}
 
double support(Rules r) {
  double s = 0;
  itemSet t = unionSet(r.lhs, r.rhs);
  int countAB = 0;
  for(int i=0; i<ROWS; i++) {
    if(isSubset(t,i))
      countAB++;
  }

  s = countAB/(double)ROWS;
  return s;
}

double confidence(Rules r) {
  double s = support(r);
  double c = 0;
  int countA = 0;
  for(int i=0; i<ROWS; i++) {
    if(isSubset(r.lhs,i))
      countA++;
  }

  c = s/((double)countA/(double)ROWS);
  return c;
}

double lift(Rules r) {
  double c = confidence(r);
  double l = 0;
  int countB = 0;
  for(int i=0; i<ROWS; i++) {
    if(isSubset(r.rhs,i))
      countB++;
  }
  
  l = c/((double)countB/(double)ROWS);
  return l;
}

void computeMeasures(Rules &r) {
  r.support = support(r);
  r.confidence = confidence(r);
  r.lift = lift(r);
}




