import ipcapture.*;
import gab.opencv.*;
import processing.video.*;
import java.io.File;
import java.awt.Rectangle;
OpenCV opencv; 
Rectangle[] faces;



GAN autoencoder; //I know it isn't a gan. This program has been through a few iterations.
//Network Judge;
//Anykey switch between training and messing around
//mouseLeft blot out face
//mouseRight fill in face
//mouseMiddle new face
PGraphics sribble;
File path = new File("C:\\Users\\Eyeba\\Desktop\\facer\\Examples"); //file path to examples
String[] images = path.list();
PImage[] loaded = new PImage[images.length];
int num = 0;
float icount = 500;
float lr = 0.05;
double[][] ins = new double[loaded.length][];
int learnCount = 10;

IPCapture cam;

/*
float Basedisclearningrate = 0.01;
float BaselearnFromDiscRate = 0.3;
float BaselearnFromSelfRate = 0.01;
*/
//float disclearningrate = 0.01;
//float learnFromDiscRate = 0.001;
float learnFromSelfRate = 0.01;

float learnFromDiscRateBASE = 0.001;
  
float runningError = 0;
 
int mode = 1;
 
double[] zero = {0};
double[] one = {1};

boolean trainGan = false;
boolean learnFromGan = false;
boolean trainAutoEncoder = true;
boolean train = false;

int imageWidth = 28;

float errorStack = 0;

boolean incol = false;

boolean yeet = false;

void yayeet() {
  //784
  autoencoder = new GAN("784,512,256,16", "256,512,784", 0.3);
}


void setup() {
  sribble = createGraphics(imageWidth, imageWidth);
  sribble.beginDraw();
  sribble.background(0);
  sribble.endDraw();
  size(420, 280);
  
  
  
  //Judge = new Network("1200,512,196,49,1", 0.4);
  
  load();
  
  yayeet();
  
  cam = new IPCapture(this, "http://192.168.100.128:8080/video", "", "");
  cam.start();
  cam.resize(width,height);
  opencv = new OpenCV(this, width, height); 
  opencv.loadCascade(OpenCV.CASCADE_FRONTALFACE);
}
/*
void rejig() {
  imageWidth *= 2;
  
  load();
  
  autoencoder.Gen = autoencoder.Gen.backGrow(imageWidth*imageWidth,1,0);
  autoencoder.Dis = autoencoder.Dis.grow(imageWidth*imageWidth,1,0);
}
*/
void load() {
  sribble = createGraphics(imageWidth, imageWidth);
  sribble.beginDraw();
  sribble.background(0);
  sribble.endDraw();
  if (yeet) {
    loaded = new PImage[images.length];
    ins = new double[loaded.length][];
    for (int i = 0; i<loaded.length; i++) {
      loaded[i] = loadImage("Examples/" + images[i]);
      loaded[i].resize(imageWidth, imageWidth);
      ins[i] = PImagetoDoubleArray(loaded[i], incol);
      println(i + "/" + (loaded.length-1));
    }
  } else {
    File path2 = new File("C:\\Users\\Eyeba\\Desktop\\facer\\Examples2"); //file path to examples
    String[] images2 = path2.list();
    loaded = new PImage[images2.length];
    ins = new double[loaded.length][];
    for (int i = 0; i<loaded.length; i++) {
      loaded[i] = loadImage("Examples2/" + images2[i]);
      loaded[i].resize(imageWidth, imageWidth);
      ins[i] = PImagetoDoubleArray(loaded[i], incol);
      println(i + "/" + (loaded.length-1));
    }
  }
}

void draw() {
  //train
  //logFC = log(learnCount);
  //learnFromDiscRate = (1 - runningError/(icount+1))*learnFromDiscRateBASE;
  if (mode == 0) {
    trainMode();
  } else if (mode == 1) {
    chillMode();
    if (mouseX < 60) optionBars();
  } if (mode == 2) {
    videoMode();
  }
  
  
  
  stroke(0);
  strokeWeight(1);
  float bar = 8;
  stroke(0);
  fill(230);
  ellipse(bar, height - bar, bar*3/4, bar*3/4);
  ellipse(bar*2, height - bar, bar*3/4, bar*3/4);
  ellipse(bar*3, height - bar, bar*3/4, bar*3/4);
  fill(230, 30, 30);
  if (mode == 0) ellipse(bar, height - bar, bar*3/4, bar*3/4);
  fill(30, 230, 30);
  if (mode == 1) ellipse(bar*2, height - bar, bar*3/4, bar*3/4);
  fill(30, 30, 230);
  if (mode == 2) ellipse(bar*3, height - bar, bar*3/4, bar*3/4);
}

void optionBars() {
  fill(60, 60, 220);
  stroke(0);
  rect(5, 5, 15, 65);
  fill(255);
  rect(5, 70, 15, -((log(1/learnFromSelfRate)/5)*65)/2);
  text(learnFromSelfRate, 10, 10);
  if (mousePressed && hitRec(5, 5, 15, 65)) {
    learnFromSelfRate = 1/pow(10, map(mouseY, 5, 70, 5, 0));
  }
  
  fill(60, 60, 220);
  stroke(0);
  rect(5, 80, 15, 65);
  fill(255);
  //rect(5, 145, 15, map(DSIndex, 0, 6, 0, -65));
  if (mousePressed && hitRec(5, 80, 15, 65)) {
  //  DSIndex = round(map(mouseY, 145, 80, 0, 6));
  //  downScale = possibleDS[DSIndex];
  }
}

boolean hitRec(int x, int y, int wid, int hei) {
  if ((mouseX > x) && (mouseX < x + wid) && (mouseY > y) && (mouseY < y + hei)) {
    return true;
  } else {
    return false;
  }
}

void videoMode() {
  if (cam.isAvailable()) {
    cam.read();
  }
  
  if (cam.width>0) {
    PImage yeetus = DoubleArraytoPImage(PImagetoDoubleArray(cam, true), cam.width, cam.height);
    yeetus.resize(width,height);
    opencv.loadImage(yeetus); 
    faces = opencv.detect();
    image(yeetus, 0, 0);
  
    if (faces!=null) { 
      for (int i=0; i< faces.length; i++) { 
        noFill(); 
        stroke(255, 255, 0); 
        strokeWeight(10); 
        rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
        
        PImage face = yeetus.get(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
        face.resize(imageWidth, imageWidth);
        double[] imgDA = PImagetoDoubleArray(face, incol);
        double[] imgDAG = new double[imgDA.length];
        double[] srub = PImagetoDoubleArray(sribble, incol);
        for (int j = 0; j<imgDAG.length; j++) {
          imgDAG[j] = max((float) imgDA[j] - (float)srub[j], 0);
        }
        
        face = DoubleArraytoPImage(autoencoder.judgeSelf(imgDAG), imageWidth, imageWidth);
        image(face, faces[i].x, faces[i].y, faces[i].width, faces[i].height);
      }
      
      if (faces.length<=0) { 
        textAlign(CENTER); 
        fill(255, 0, 0); 
        textSize(56);
        text("UNDETECTED", 200, 100);
      }
    }   
  }
}

void chillMode() {
  double inx = (float(mouseX)/float(width)-0.5)*5;
  double iny = (float(mouseY)/float(height)-0.5)*5;
  double[] in = {inx,iny,sin(float(frameCount)/30)*2,cos(float(frameCount)/40)*2,sin(float(frameCount)/50)*2,cos(float(frameCount)/70)*2,sin(float(frameCount)/90)*2,cos(float(frameCount)/85)*2,sin(float(frameCount)/130)*2,cos(float(frameCount)/151)*2,sin(float(frameCount)/171)*2,cos(float(frameCount)/210)*2,sin(float(frameCount)/30)*2,cos(float(frameCount)/40)*2,sin(float(frameCount)/50)*2,cos(float(frameCount)/70)*2,sin(float(frameCount)/90)*2,cos(float(frameCount)/85)*2,sin(float(frameCount)/130)*2,cos(float(frameCount)/151)*2,sin(float(frameCount)/171)*2,cos(float(frameCount)/210)*2,sin(float(frameCount)/30)*2,cos(float(frameCount)/40)*2,sin(float(frameCount)/50)*2,cos(float(frameCount)/70)*2,sin(float(frameCount)/90)*2,cos(float(frameCount)/85)*2,sin(float(frameCount)/130)*2,cos(float(frameCount)/151)*2,sin(float(frameCount)/171)*2,cos(float(frameCount)/210)*2};
  //double[] in = {inx,iny,sin(float(frameCount)/30)*2,cos(float(frameCount)/40)*2,sin(float(frameCount)/50)*2,cos(float(frameCount)/70)*2,sin(float(frameCount)/90)*2,cos(float(frameCount)/85)*2};
  image(DoubleArraytoPImage(autoencoder.judge(in), imageWidth, imageWidth), 0, 0, width-140, height);
  if (mousePressed) {
    sribble.beginDraw();
    if (mouseButton == RIGHT) {
      sribble.fill(0);
      sribble.stroke(0);
      sribble.ellipse(map(mouseX, 280, 420, 0, imageWidth), map(mouseY, 0, 140, 0, imageWidth), 3, 3);
    } else if (mouseButton == LEFT) {
      sribble.fill(255);
      sribble.stroke(255);
      sribble.ellipse(map(mouseX, 280, 420, 0, imageWidth), map(mouseY, 0, 140, 0, imageWidth), 3, 3);
    } else {
      num = floor(random(loaded.length));
    }
    sribble.endDraw();
  }
  sideFaces();
}

void sideFaces() {
  PImage img = loaded[num];
  double[] imgDA = PImagetoDoubleArray(img, incol);
  double[] imgDAG = new double[imgDA.length];
  double[] srub = PImagetoDoubleArray(sribble, incol);
  for (int j = 0; j<imgDAG.length; j++) {
      imgDAG[j] = max((float) imgDA[j] - (float)srub[j], 0);
  }
  image(DoubleArraytoPImage(imgDAG, imageWidth, imageWidth), width-140, 0, 140, 140);
  image(DoubleArraytoPImage(autoencoder.judgeSelf(imgDAG), imageWidth, imageWidth), width-140, 140, 140, 140);
}

void trainMode() {
  train(floor(icount));
  //println(runningError/(icount+1));
  runningError= 0;
  for (int i = 0; i<width-140; i+=28) {
    for (int j = 0; j<height; j+=28) {
      double inx = (float(i)/float(width) - 0.5)*25;
      double iny = (float(j)/float(height)-0.5)*25;
      double[] in = {inx,iny,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
      image(DoubleArraytoPImage(autoencoder.judge(in), imageWidth, imageWidth), i, j, 28, 28);
    }
  }
  num = floor(random(loaded.length));
  sideFaces();
}

void train(int iterations) {
  for (int lol = 0; lol<iterations; lol++) {
    num = floor(random(loaded.length));
    double[] imgDA;
    imgDA = ins[num];

    double[] imgDAG = new double[imgDA.length]; 
    
    /*
    for (int j = 1; j<imageWidth-1; j++) {
      for (int k = 1; k<imageWidth-1; k++) {
        imgDAG[j + k*imageWidth] = (imgDA[j - 1 + k*imageWidth] + imgDA[j + 1 + k*imageWidth] + imgDA[j + (k+1)*imageWidth] + imgDA[j + (k-1)*imageWidth])/4;
      }
    }*/
    
    for (int i = 0; i<imgDAG.length; i++) {
      imgDAG[i] = imgDA[i];
    }
    
    int rounds = floor(random(5));
    
    for (int i = 0; i<rounds; i++) {
      float wid = random(imageWidth/6);
      float hei = random(imageWidth/6);
      float x = random(imageWidth);
      float y = random(imageWidth);
      for (int j = 0; j<imageWidth; j++) {
        for (int k = 0; k<imageWidth; k++) {
          if ((j > x) && (j<x+wid) && (k>y) && (k<y+hei)) {
            imgDAG[j + k*imageWidth] = 0;
          }
        }
      }
    }
    /*
    if (rounds <= 1) {
      for (int i = 0; i<imgDAG.length; i++) {
        if (random(2) > 0.5) imgDAG[i] = (sign(random(-1, 1))*0.5) + 0.5;
      }
    }
    */
    autoencoder.trainThroughBoth(imgDAG, imgDA, learnFromSelfRate);
    //trainApart();
  //Now the autoencoder has the right neurons
  //double[] autothoughts = autoencoder.Dis.neurons[autoencoder.Dis.layers-1];
  /*
  if (learnFromGan) {
    double[] out = autoencoder.Gen.neurons[autoencoder.Gen.layers - 1];
    autoencoder.Gen.learn(imgDAG, out, learnFromSelfRate/10);
    
    autoencoder.net.learnFromErrorExt(imgDA, Judge.getLayerError(autothoughts, zero, 0, true, true), autoencoder.disStart, learnFromDiscRate, false);
    autoencoder.net.learnFromErrorExt(imgDA, Judge.getLayerError(autothoughts, one, 0, false, true), autoencoder.disStart, learnFromDiscRate, false);
  }
  
  
  if (trainGan) {
    Judge.learn(autothoughts, zero, disclearningrate);
    Judge.learn(imgDA, one, disclearningrate);
  }
    */
    //autoencoder.trainBoth(imgDAG, imgDA, 0.01);
    //double[] thought = autoencoder.judgeSelf(imgDAG);
    //double[] zero = {0};
    //double[] one = {1};
    //Judge.learn(thought, zero, 0.01);
    //Judge.learn(imgDA, one, 0.01);
    //autoencoder.net.learnFromError(imgDA, Judge.getLayerError(thought, one, 0, false), 0.01);
    //runningError += Judge.lastError;
    if (lol == icount-1) learnCount++;
  }
}
/*
void trainApart() {
  int num1 = floor(random(ins.length));
  int num2 = floor(random(ins.length));
  double[] yeet1 = autoencoder.Gen.think(ins[num1]);
  double[] yeet2 = autoencoder.Gen.think(ins[num2]);
  double[] error = new double[yeet1.length];
  for (int i = 0; i<yeet1.length; i++) error[i] = -(yeet2[i] - yeet1[i]);
  autoencoder.Gen.learnFromError(error, learnFromSelfRate);
}*/

void keyReleased() {
  if (key == ' ') {
    train = !train;
  } else if (key == '2') {
    mode = 1;
  } else if (key == '1') {
    mode = 0;
  } else if (key == '3') {
    mode = 2;
  } else if (key == 'k') {
    yayeet();
  } else if (key == 'h') {
    yeet = !yeet;
    load();
  }
  num = floor(random(loaded.length));
}
class Network {
  double[][][] weights;
  double[][] bias;
  double[][] neurons; //values
  int layers;
  String nstr;
  boolean[] noSigonLayer;
  int[] layerSizes;
  float lastError = 0;
  Network(String layerSizesz, float randomize, float x) {
    nstr = layerSizesz;
    String[] layerSizesstr = split(layerSizesz, ',');
    layers = layerSizesstr.length;
    layerSizes = new int[layerSizesstr.length];
    for (int i = 0; i<layerSizesstr.length; i++) {
      layerSizes[i] = int(layerSizesstr[i]);
    }
    printArray(layerSizes);
    randomize = abs(randomize);
    noSigonLayer = new boolean[layers];
    
    weights = new double[layerSizes.length][][]; //Make the network the right size.
    neurons = new double[layerSizes.length][];
    bias = new double[layerSizes.length][];
    neurons[0] = new double[layerSizes[0]];
    for (int i = 1; i<layerSizes.length; i++) {
      noSigonLayer[i] = false;
      weights[i] = new double[layerSizes[i]][layerSizes[i-1]];
      neurons[i] = new double[layerSizes[i]];
      bias[i] = new double[layerSizes[i]];
    }
    
    //randomise Weights
    for (int i = 1; i<layerSizes.length; i++) {
      for (int j = 0; j<layerSizes[i]; j++) {
        for (int k = 0; k<layerSizes[i-1]; k++) weights[i][j][k] = random(-randomize, randomize) + x;
        bias[i][j] = random(-randomize, randomize);
      }
    }
    
  }
  
  Network grow(int newNeurons, float randomize, float x) {
    randomize = abs(randomize);
    Network temp = new Network(nstr + "," + newNeurons, randomize, x);
    
    for (int i = 1; i<temp.layers-1; i++) {
      for (int j = 0; j<temp.layerSizes[i]; j++) {
        for (int k = 0; k<temp.layerSizes[i-1]; k++) {
          temp.weights[i][j][k] = weights[i][j][k];
        }
        temp.bias[i][j] = bias[i][j];
      }
      temp.noSigonLayer[i] = noSigonLayer[i];
    }
    
    return temp;
  }
  
  Network backGrow(int newNeurons, float randomize, float x) {
    randomize = abs(randomize);
    Network temp = new Network(newNeurons + "," + nstr, randomize, x);
    
    for (int i = 2; i<temp.layers; i++) {
      for (int j = 0; j<temp.layerSizes[i]; j++) {
        for (int k = 0; k<temp.layerSizes[i-1]; k++) {
          temp.weights[i][j][k] = weights[i-1][j][k];
        }
        temp.bias[i][j] = bias[i-1][j];
      }
      temp.noSigonLayer[i] = noSigonLayer[i-1];
    }
    
    return temp;
  }
  
  double[] think(double inputs[]) {
    return thinkExt(inputs, 0, layers-1);
  }
  
  double[] thinkExt(double inputs[], int thinkFrom, int thinkTo) {

    for (int i = 0; i<neurons.length; i++) {
      neurons[thinkFrom][i] = inputs[i];
    }
    
    double[] outPuts = new double[layerSizes[thinkTo]];
    for (int i = thinkFrom+1; i<=thinkTo; i++) {//layer
      for (int k = 0; k<layerSizes[i]; k++) {
        neurons[i][k] = bias[i][k];
        for (int j = 0; j<layerSizes[i-1]; j++) neurons[i][k] += weights[i][k][j]*neurons[i-1][j];
        if (noSigonLayer[i]) {} else {neurons[i][k] = sigmoid(neurons[i][k]);}
      }
    }
    
    outPuts = neurons[thinkTo];
    
    return outPuts;
  }
  
  double[] learn(double[] inputs, double[] desiredOutputs, float stepSize) {
    double[] error = new double[desiredOutputs.length];
    double[] played = think(inputs);
    for (int i = 0; i<error.length; i++) {
      double E = (played[i] - desiredOutputs[i]);
      error[i] = E;
    }
    return learnFromError(error, stepSize);
  }
  
  double[] getLayerError(double[] inputs, double[] desiredOutputs, int layerNum, boolean negative) {
    double[][] error = new double[neurons.length][];
    double[] played = think(inputs);
    
    for (int i = 0; i<error.length; i++) {
      error[i] = new double[neurons[i].length];
      for (int j = 0; j<error[i].length; j++) error[i][j] = (double) 0;
    }
    
    for (int i = 0; i<layerSizes[layerSizes.length-1]; i++) {
      error[layerSizes.length-1][i]=(played[i] - desiredOutputs[i])*dsigmoid(neurons[layers-1][i]);
    }
    
    for (int i = layers-2; i>0; i--) {
      for (int j = 0; j<layerSizes[i]; j++) {
        double sum = 0;
        for (int k = 0; k< error[i+1].length; k++) {
          sum += weights[i+1][k][j] * error[i+1][k];
        }
        if (noSigonLayer[i]) {error[i][j] = sum*neurons[i][j];} else {error[i][j] = sum*dsigmoid(neurons[i][j]);}
      }
    }
    
    double[] out = new double[error[layerNum].length];
    if (layerNum == 0) {
      for (int j = 0; j<layerSizes[0]; j++) {
        double sum = 0;
        for (int k = 0; k<error[1].length; k++) {
          sum += weights[1][k][j] * error[1][k];
        }
        out[j] = sum*inputs[j];
      }
    } else {
      for (int j = 0; j<layerSizes[layerNum]; j++) out[j] = error[layerNum][j];
    }
    if (negative) for (int i = 0; i<out.length; i++) out[i] *= -1;
    
    return out;
  }

  double[] learnFromError(double[] inError, float stepSize) {
    double[][] error = new double[neurons.length][];
    
    for (int i = 0; i<error.length; i++) {
      error[i] = new double[neurons[i].length];
      for (int j = 0; j<error[i].length; j++) error[i][j] = (double) 0;
    }
    
    for (int i = 0; i<layerSizes[layerSizes.length-1]; i++) {
          if (noSigonLayer[layers - 1]) {
            error[layerSizes.length-1][i]=inError[i];
          } else {
            error[layerSizes.length-1][i]=inError[i]*dsigmoid(neurons[layers-1][i]);
          }
    }
    
    for (int i = layers-2; i>0; i--) {
      for (int j = 0; j<layerSizes[i]; j++) {
        double sum = 0;
        for (int k = 0; k< error[i+1].length; k++) {
          sum += weights[i+1][k][j] * error[i+1][k];
        }
        if (noSigonLayer[i]) {error[i][j] = sum*neurons[i][j];} else {error[i][j] = sum*dsigmoid(neurons[i][j]);}
      }
    }
    
    //tweak weights
    for (int i = 1; i<layers; i++) {
      for (int j = 0; j<neurons[i].length; j++) {
        for (int pj = 0; pj<neurons[i-1].length; pj++) {
          double delta = -stepSize * neurons[i-1][pj] * error[i][j];
          weights[i][j][pj] += delta;
        }
        bias[i][j] += -stepSize * error[i][j];
      }
    }
    
    double[] out = new double[neurons[0].length];
      for (int j = 0; j<layerSizes[0]; j++) {
        double sum = 0;
        for (int k = 0; k<error[1].length; k++) {
          sum += weights[1][k][j] * error[1][k];
        }
        out[j] = sum;
      }
      
      return out;
  }
                                                             //0          1            layers
  void learnExt(double[] inputs, double[] desiredOutputs, int in, int learnFrom, int learnTo, float stepSize) {
    double[][] error = new double[neurons.length][];
    double[] played = thinkExt(inputs, in, layers-1);
    
    for (int i = 0; i<error.length; i++) {
      error[i] = new double[neurons[i].length];
      for (int j = 0; j<error[i].length; j++) error[i][j] = (double) 0;
    }
    float totalError = 0;
    for (int i = 0; i<layerSizes[layerSizes.length-1]; i++) {
      error[layerSizes.length-1][i]=(played[i] - desiredOutputs[i])*dsigmoid(neurons[layers-1][i]);
      totalError += abs((float) error[layerSizes.length-1][i]);
    }
    lastError= totalError/float(layerSizes[layerSizes.length-1]);
    //println(error[layers-1][0]); //<- Print error?
    
    for (int i = layers-2; i>learnFrom; i--) {
      for (int j = 0; j<layerSizes[i]; j++) {
        double sum = 0;
        for (int k = 0; k< error[i+1].length; k++) {
          sum += weights[i+1][k][j] * error[i+1][k];
        }
        if (noSigonLayer[i]) {error[i][j] = sum*neurons[i][j];} else {error[i][j] = sum*dsigmoid(neurons[i][j]);}
      }
    }
    
    //tweak weights
    for (int i = learnFrom; i<learnTo; i++) {
      for (int j = 0; j<neurons[i].length; j++) {
        for (int pj = 0; pj<neurons[i-1].length; pj++) {
          double delta = -stepSize * neurons[i-1][pj] * error[i][j];
          weights[i][j][pj] += delta;
        }
        bias[i][j] += -stepSize * error[i][j];
      }
    }
    
  }
  
}
  
float sign(float in) {if (in>0) {return 1;} else {return -1;}}

double sigmoid(double input) { return (float(1) / (float(1) + Math.pow(2.71828182846, -input))); }
double dsigmoid(double input) { return (input*(1-input)); }

double[] PImagetoDoubleArray(PImage img, boolean incolor) {
  double[] out;
  if (incolor) {out = new double[img.pixels.length*3];} else {out = new double[img.pixels.length];}
  img.loadPixels();
  if (incolor) {
    for (int i = 0; i<img.pixels.length;i++) {
      out[i*3] = red(img.pixels[i])/255;
      out[i*3 + 1] = green(img.pixels[i])/255;
      out[i*3 + 2] = blue(img.pixels[i])/255;
    }
  } else {
    for (int i = 0; i<img.pixels.length;i++) {
      out[i] = (red(img.pixels[i]) + green(img.pixels[i]) + blue(img.pixels[i]))/(3*255);
    }
  }
  img.updatePixels();
  return out;
}

PImage DoubleArraytoPImage(double[] in, int wid, int hei) {
  PImage img;
  boolean incolor = !(wid*hei == in.length);
  img = new PImage(wid, hei);
  img.loadPixels();
  if (incolor) {
    for (int i = 0; i<img.pixels.length;i++) {
      img.pixels[i] = color((float) in[i*3]*255,(float) in[i*3 + 1]*255,(float) in[i*3 + 2]*255);
    }
  } else {
    for (int i = 0; i<img.pixels.length;i++) {
      img.pixels[i] = color((float) in[i]*255);
    }
  }
  img.updatePixels();
  return img;
}

class GAN {
  Network net;
  int disStart;
  GAN(String gen, String add, float ran) {
    String network = gen + "," + add;
    disStart = split(gen, ',').length;
    net = new Network(network, ran,0);
    //net.noSigonLayer[disStart-1] = true;
    println(disStart);
  }
  
  void trainThroughBoth(double[] in, double[] out, float lr) {
    net.learnExt(in, out, 0, 1, net.layers, lr);
  }
  
  void trainAdd(double[] in, double[] out, float lr) {
    net.learnExt(in, out, disStart-1, disStart, net.layers, lr);
  }
  
  void trainGen(double[] in, double[] out, float lr) {
    net.learnExt(in, out, 0, 1, disStart-1, lr);
  }
  
  double[] think(double[] in) {
    return net.thinkExt(in, 0, disStart-1);
  }
  
  double[] judge(double[] in) {
    return net.thinkExt(in, disStart-1, net.layers-1);
  }
  
  double[] judgeSelf(double[] in) {
    return net.thinkExt(in, 0, net.layers-1);
  }
}
