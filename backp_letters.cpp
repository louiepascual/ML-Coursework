/**
  ICST347 Assignment #6
  backp_letters.cpp

  Purpose: 
  Uses the backpropagation algorithm to learn the weights of the character recognition ANN.

  @original_author Allan A. Sioson, PhD. 

  Reimplemented for Character Recognition ANN by:
  @author Arazas, Al Jamil L. (2012-1-0712)
  @author Pascual, Louie Lester E. (2013-1-1743)
  @author Valeros, John Dinnex M. (2013-1-1618)

  @version 1.0 09/05/2016

  In our honor, we assure that we have not given nor received any
  unauthorized help in this work.

    Al Jamil L. Arazas, Louie Lester E. Pascual, John Dinnex M. Valeros


  Answer to questions:
  1) a. Input Units = 26, Hidden Units = 5, Output Units = 26
     b. Using 10000 Iterations, it converged on 6485th epoch.
     c. By checking if each weight difference between epochs was around the -1e-3 to 1e-3 range.
     d. Yes, 100% accuracy of each letter was achieved when 10000 epochs was used.
     e. 
      E = (Predicted: E, F & K) 
          1 1 1 0 0
          1 0 0 0 0
          1 1 1 0 0
          1 0 0 0 0
          1 1 1 0 0
      L = (Predicted: L, N, & U)
          1 0 0 0 0
          1 0 0 0 0
          1 0 0 0 0
          1 0 0 0 0
          1 1 1 0 0     
      I = (Predicted: B, I &, T)
          0 0 1 0 0
          0 0 1 0 0
          0 0 1 0 0
          0 0 1 0 0
          0 0 1 0 0
      V = (Predicted: Q & V)
          1 0 0 0 1
          1 0 0 0 1
          1 0 0 0 1
          0 1 0 1 1
          0 1 1 1 0
      O = (Predicted: O & P)
        0 1 1 1 0
        1 0 0 0 1
        1 0 0 0 1
        1 0 0 0 1
        0 1 1 1 0


*/
# include <iostream>
# include <ctime>
# include <cstdlib>
# include <map>
# include <vector>
# include <cmath>

# define ITERS 10000

/* N nodes, NT targets, NH hidden, NI inputs */
# define N 57
# define NT 26  
# define NH 5
# define NI 26

using namespace std;

struct data {
  double x[NI];
  double t[NT];
};

data D[] = {
  { { 1,
	  0, 1, 1, 1, 0,                    
	  0, 1, 0, 1, 0,
	  1, 1, 1, 1, 1, 
	  1, 0, 0, 0, 1,
	  1, 0, 0, 0, 1},{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //A
  {  {1,
	  1, 1, 1, 1, 0,
	  1, 0, 0, 1, 0,
	  1, 1, 1, 1, 1, 
	  1, 0, 0, 0, 1,
	  1, 1, 1, 1, 1},{0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //B
  {  {1,
	  1, 1, 1, 1, 0,
	  1, 0, 0, 1, 0,
	  1, 0, 0, 0, 0,
	  1, 0, 0, 0, 1,
	  1, 1, 1, 1, 1},{0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //C
  {  {1,
	  1, 1, 1, 1, 0,
	  1, 0, 0, 1, 1,
	  1, 0, 0, 0, 1,
	  1, 0, 0, 0, 1,
	  1, 1, 1, 1, 1},{0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //D
  { {1,
	 1, 1, 1, 1, 1,
	 1, 0, 0, 0, 0,
	 1, 1, 1, 1, 0,
	 1, 0, 0, 0, 0,
	 1, 1, 1, 1, 1},{0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //E

  {  {1,
	  1, 1, 1, 1, 1,
	  1, 0, 0, 0, 0,
	  1, 1, 1, 1, 0,
	  1, 0, 0, 0, 0,
	  1, 0, 0, 0, 0},{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //F
  { {1,
	 1, 1, 1, 1, 0,
	 1, 0, 0, 0, 0,
	 1, 0, 0, 1, 1,
	 1, 0, 0, 0, 1,
	 1, 1, 1, 1, 1} ,{0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //G
  { {1,
	 1, 0, 0, 0, 1,
	 1, 0, 0, 0, 1,
	 1, 1, 1, 1, 1,
	 1, 0, 0, 0, 1,
	 1, 0, 0, 0, 1} ,{0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //H
  { {1,
	 0, 1, 1, 1, 0,
	 0, 0, 1, 0, 0,
	 0, 0, 1, 0, 0,
	 0, 0, 1, 0, 0, 
	 0, 1, 1, 1, 0} ,{0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //I
  {  {1,
	  0, 0, 0, 0, 1,
	  0, 0, 0, 0, 1,
	  0, 0, 0, 0, 1,
	  1, 0, 0, 0, 1,
	  1, 1, 1, 1, 1}, {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //J
  {  {1,
   	  1, 0, 0, 0, 1, 
	  1, 0, 0, 1, 0, 
	  1, 1, 1, 0, 0,
	  1, 0, 0, 1, 0,
	  1, 0, 0, 0, 1} ,{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //K
  {  {1,
      1, 0, 0, 0, 0,
	  1, 0, 0, 0, 0,
	  1, 0, 0, 0, 0,
	  1, 0, 0, 0, 0,
	  1, 1, 1, 1, 1},{0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //L
  { {1,
     1, 0, 0, 0, 1,
	 1, 1, 0, 1, 1,
	 1, 0, 1, 0, 1,
	 1, 0, 1, 0, 1,
	 1, 0, 0, 0, 1},{0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}}, //M
  { {1, 
 	 1, 0, 0, 0, 1,
 	 1, 1, 0, 0, 1,
 	 1, 0, 1, 0, 1,
 	 1, 0, 0, 1, 1,
 	 1, 0, 0, 0, 1},{0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}}, //N
  { {1,
     1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}}, //O
  { {1,
     1, 1, 1, 1, 1,
	 1, 0, 0, 0, 1,
	 1, 1, 1, 1, 1,
	 1, 0, 0, 0, 0,
	 1, 0, 0, 0, 0},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}}, //P
  {  {1,
      1, 1, 1, 1, 1,
	  1, 0, 0, 0, 1,
	  1, 0, 0, 0, 1,
	  1, 0, 0, 1, 1,
	  1, 1, 1, 1, 1},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}}, //Q
  {  {1,
	  1, 1, 1, 1, 1,
	  1, 0, 0, 0, 1,
	  1, 1, 1, 1, 1, 
	  1, 0, 0, 1, 0,
	  1, 0, 0, 0, 1},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}}, //R
  { {1,
     1, 1, 1, 1, 1,
	 1, 0, 0, 0, 0,
	 1, 1, 1, 1, 1, 
	 0, 0, 0, 0, 1,
	 1, 1, 1, 1, 1},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}}, //S  
  {  {1,
	  1, 1, 1, 1, 1,
	  1, 0, 1, 0, 1,
	  0, 0, 1, 0, 0,
	  0, 0, 1, 0, 0,
	  0, 1, 1, 1, 0},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}}, //T
  {  {1,
	  1, 0, 0, 0, 1,
	  1, 0, 0, 0, 1,
	  1, 0, 0, 0, 1,
	  1, 0, 0, 0, 1,
	  1, 1, 1, 1, 1},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}}, //U
  { {1,
	 1, 0, 0, 0, 1,
	 1, 0, 0, 0, 1,
	 0, 1, 0, 1, 0,
	 0, 1, 0, 1, 0,
	 0, 1, 1, 1, 0},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}}, //V
  { {1,
	 1, 0, 0, 0, 1,
	 1, 0, 0, 0, 1,
	 1, 0, 1, 0, 1,
	 1, 0, 1, 0, 1,
	 1, 1, 1, 1, 1},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}}, //W
  { {1,
	 1, 0, 0, 0, 1,
	 0, 1, 0, 1, 0,
	 0, 0, 1, 0, 0,
	 0, 1, 0, 1, 0,
	 1, 0, 0, 0, 1},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}}, //X
  {  {1,
	  1, 0, 0, 0, 1,
	  1, 0, 0, 0, 1,
	  1, 1, 1, 1, 1,
	  0, 0, 1, 0, 0,
	  0, 0, 1, 0, 0},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}}, //Y
  {  {1,
	  1, 1, 1, 1, 1,
	  0, 0, 0, 1, 1,
	  0, 1, 1, 1, 0,
	  1, 1, 0, 0, 0,
	  1, 1, 1, 1, 1},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}} //Z
};

int numRecs = 26;

double sigmoid(double y) {
  return (1.0 / (1.0 + exp(-y)));
}

void computeNetwork(double w[][N], double x[]) {

  // compute the units in the hidden layer
  for(int i=NI; i<NI+NH; i++) {
    double s=0.0;

    // traverse through all the nodes in input layer
    for(int j=0; j<NI; j++) {
      s += w[i][j] * x[j];
    }
    x[i] = sigmoid(s);
  }

  // compute the units in the target layer
  for(int i=NI+NH; i<N; i++) {
    double s=0.0;

    // traverse through all the nodes in hidden layer
    for(int j=NI; j<NI+NH; j++) {
      s += w[i][j] * x[j];
    }
    x[i] = sigmoid(s);
  }
}

void setRandom(double w[][N]) {
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) 
      w[j][i] = double((rand() % 201) - 100) / 1000;
}

void setZero(double w[][N]) {
  for (int j = 0; j < N; j++) 
    for (int i = 0; i < N; i++)
      w[j][i] = 0.0;
}

void backpropagate(double h, double w[][N]) {
  double x[N];
  double dw[N][N];
  double t[NT];
  double delta[N];
  double alpha = 0.5;
  setRandom(w);
  for (int z = 0; z < ITERS; z++) {
    for (int r = 0; r < numRecs; r++) {
      // initalize compute units
      for (int i = 0; i < NI; i++)
        x[i] = D[r].x[i];
        
      // initialize target units
      for (int i = 0; i < NT; i++)
        t[i] = D[r].t[i];

      // 1. process compute units
      computeNetwork(w,x);

      // 2. compute delta of output units
      for(int i=NI+NH, j=0; i<N; i++,j++) {
        delta[i] = x[i] * (1-x[i]) * (t[j] - x[i]);
      }

      // 3. compute delta of hidden units
      for(int i=NI; i<NI+NH; i++) {
        double s = 0.0;
        for(int j=NI+NH; j<N; j++) {
          s += w[j][i] * delta[j];
        }
        delta[i] = x[i] * (1 - x[i]) * s;
      }

      // 4. update the weights
      for (int j = 0; j < N; j++) 
        for (int i = 0; i < N; i++) {
          dw[j][i] = h * delta[j] * x[i] + (alpha * dw[j][i]);
          w[j][i]  = w[j][i] + dw[j][i]; 
        }
    }
  }
}

int main() {  
  // initialize weight array
  double w[N][N];
  backpropagate(0.3, w);

  double x[N];
  x[0] = 1.0;

  while (true) {
    // get input data
    cout << ":: ";
    cin >> x[1] >> x[2] >> x[3] >> x[4] >> x[5];
    cin >> x[6] >> x[7] >> x[8] >> x[9] >> x[10]; 
    cin >> x[11] >> x[12] >> x[13] >> x[14] >> x[15];
    cin >> x[16] >> x[17] >> x[18] >> x[19] >> x[20]; 
    cin >> x[21] >> x[22] >> x[23] >> x[24] >> x[25];
    computeNetwork(w,x);

    // Output Results
    cout << " =";
    for (int i = NI+NH; i < N; i++)
      cout << " " << x[i];
    cout << endl;
    cout << " =";
    
    for (int i = NI+NH; i < N; i++) {
      cout << " " << round(x[i]);  
    }
    cout << endl;
  }
}






