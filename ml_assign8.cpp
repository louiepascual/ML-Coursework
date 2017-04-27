/**
  Assignment #8: Detecting face orientation using a 1-layer feedforward ANN
  ml_assign8.cpp

  Purpose:
  Implement a sigmoid-based 1-layer feedforward Artificial Neural Network 
  that recognizes the orientation of the face of a person in an image.
  
  @author Arazas, Al Jamil L. (2012-1-0712)
  @author Pascual, Louie Lester E. (2013-1-1743)
  @author Valeros, John Dinnex M. (2013-1-1618)

  @version 1.0 10/09/2016

  In our honor, we assure that we have not given nor received any
  unauthorized help in this work.

    Al Jamil L. Arazas, Louie Lester E. Pascual, John Dinnex M. Valeros
*/
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
using namespace std;

#define ITERS 100

/* N nodes, NT targets, NH hidden, NI inputs */
#define N 1265
#define NT 4
#define NI 961
#define NH 300

#define numRecs 260
#define nTests 364

using namespace std;

struct data {
  double x[NI];
  double t[NT];
};

data D[numRecs];

/**
  Using the weights of the edges connecting the input nodes
  and the hidden nodes, recreate the image at a certain epoch

  @param w, the weights of the network
  @param hnode, the id of the hidden node
  @param epoch, a string that states on what epoch it is in, 
        will be used for file name
  @return none
*/
void recreateImage(double w[][N], int hnode, string epoch) {
  // fix the filename
  ofstream fout;
  char fileName[256];
  sprintf(fileName,"h%d_%s.out",hnode,epoch.c_str());
  fout.open(fileName);

  double a,b,y;
  int i;

  // hnode serves like the id of that hidden node on the hidden layer
  // x961 has an hnode of 0, x962 has 1, x963 has 2
  i = NI + hnode; 

  // search for the min and max weights
  a = 9999;
  b = -9999;
  for(int j=1; j<NI; j++) {
    a = min(w[j][i],a);
    b = max(w[j][i],b);
  }

  // proportionally recalibrate the weights and output to file
  for(int j=1; j<NI; j++) {
    y = 255 * ((w[j][i] - a)/(b - a));
    fout << y << " ";
  }
  fout.close();
}


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

void backpropagate(double h, double w[][N]) {
  double x[N];
  static double dw[N][N];
  double t[NT];
  double delta[N];
  double alpha = 0.3;
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
          dw[j][i] = (h * delta[j] * x[i]) + (alpha * dw[j][i]);
          w[j][i]  = w[j][i] + dw[j][i]; 
        }
    }

    // Output how many epochs are done
    cout << "FINISHED ITER #" << z+1 << endl;

    //Recreate the image using weights (on epoch 1 and 100)
    if(z == 0) {
      for(int i=0; i<3; i++)
        recreateImage(w,i,"001");
      cout << "DONE RECREATING IMAGE (Epoch1)\n";
    }
    else if(z == 99) {
      for(int i=0; i<3; i++)
        recreateImage(w,i,"100");

      cout << "DONE RECREATING IMAGE (Epoch100)\n";
    }
  }
}

int main(int argc, char **argv) {  
  // Check command usage
  if (argc != 3) {
      cerr << "usage: " << argv[0] << " trainingFile testFile" << endl;
      return -1;
    }
  
  // Setup input files
  string trainingFile, testFile;
  trainingFile = argv[1];
  testFile = argv[2];

  // Get and Set Input Training Data
  ifstream fin;
  fin.open(trainingFile.c_str());

  for(int i=0; i<numRecs; i++) {
    D[i].x[0] = 1.0; // bias unit

    // get attributes
    for(int j=1; j<NI; j++)
      fin >> D[i].x[j];

    // get target attribute
    int tAttr;
    fin >> tAttr;
    D[i].t[tAttr] = 1;
  }
  fin.close();

  // initialize weight array
  static double w[N][N];
  backpropagate(0.3, w);

  // Setup Test Data
  fin.open(testFile.c_str());

  int numCorrect = 0, numIncorrect = 0;
  int confusionMatrix[4][4] = {0};
  double x[N];
  for(int i=0; i<nTests; i++) {
    // Reset values of x
    memset(x,0.0,sizeof(x));
    x[0] = 1.0; // initialize bias unit
    
    // First, get the attributes
    for(int j=1; j<NI; j++)
      fin >> x[j];

    // Then, get the id of the face orientation
    int actual;
    fin >> actual;

    // Run through the network
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

    // check if predicition is correct
    double maxVal = x[NI+NH];
    int prediction = 0;
    for(int j=NI+NH+1; j<N; j++) {
      if(x[j] > maxVal) {
        prediction = j-(NI+NH);
        maxVal = x[j];
      }
    }

    cout << "Prediction:" << prediction;
    cout << "; Actual: " << actual << endl;
    if(prediction == actual)
      numCorrect++;
    else
      numIncorrect++;

    confusionMatrix[prediction][actual]++;
  }
  
  // Output the Number of Correct, Incorrect and Accuracy
  cout << "Correct Instances: " << numCorrect << endl;
  cout << "Incorrect Instances: " << numIncorrect << endl;
  cout << "Accuracy: ";
  cout << (numCorrect/(double)nTests) * 100 << "%\n";

  // Output the confusion matrix
  cout << "Confusion Matrix: \n";
  cout << "  | 0\t1\t2\t3\n";
  cout << "-------------------------------\n";
  for(int i=0; i<4; i++) {
    cout << i << " | ";
    for(int j=0; j<4; j++) {
      cout << confusionMatrix[i][j] << "\t";
    }
    cout << endl;
  }
  
}
