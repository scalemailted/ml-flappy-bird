var _createClass = function () {function defineProperties(target, props) {for (var i = 0; i < props.length; i++) {var descriptor = props[i];descriptor.enumerable = descriptor.enumerable || false;descriptor.configurable = true;if ("value" in descriptor) descriptor.writable = true;Object.defineProperty(target, descriptor.key, descriptor);}}return function (Constructor, protoProps, staticProps) {if (protoProps) defineProperties(Constructor.prototype, protoProps);if (staticProps) defineProperties(Constructor, staticProps);return Constructor;};}();function _classCallCheck(instance, Constructor) {if (!(instance instanceof Constructor)) {throw new TypeError("Cannot call a class as a function");}}var SCREEN_SIZE = Math.min(600, window.innerWidth);
var TOTAL_BIRDS = 300;
//var imgSrc = '../assets/bird.png';
var imgSrc = 'https://raw.githubusercontent.com/scalemailted/ml-flappy-bird/main/assets/bird.png';
var birds = [];
var oldGenerationBirds = [];
var pipes = [];
var highest = 0;
var counter = 0;
var generation = 1;
var slider = void 0;
var img = void 0;

function preload() {
  img = loadImage(imgSrc);
}

function setup() {
  createCanvas(SCREEN_SIZE, 3 * SCREEN_SIZE / 4);
  createP('Use the slider to increase the speed of learning:');
  slider = createSlider(1, 100, 1, 1);
  generateBirds(true);
}

function draw() {

  // Code logic
  for (var i = 0; i < slider.value(); i++) {
    // Only push new pipe at 50 frames
    if (counter % 50 === 0) {
      pipes.push(new Pipe());
    }

    // Birds functions
    birds.forEach(function (bird) {
      bird.think(pipes);
      bird.update();

      // Save highest score
      if (bird.score > highest) {
        highest = bird.score;
      }
    });

    // Pipes functions
    for (var _i = pipes.length - 1; _i >= 0; _i--) {
      var pipe = pipes[_i];
      pipe.update();

      if (pipe.isOffscreen()) {
        pipes.splice(_i, 1);
      }

      for (var j = birds.length - 1; j >= 0; j--) {
        var bird = birds[j];

        // Kill bird if hits the pipe
        // or hits the ceiling
        // or hits the floor
        if (bird.hitsPipe(pipe) ||
        bird.y - bird.r < 0 ||
        bird.y + bird.r > height) {
          oldGenerationBirds.push(birds.splice(j, 1)[0]);
        }
      }
    }

    // Increase counter
    counter++;

    checkGenerationEnd();
  }

  // Drawing
  background('#00a8ff');

  // Draw the birds
  birds.forEach(function (bird) {
    bird.draw();
  });

  // Draw the pipes
  pipes.forEach(function (pipe) {
    pipe.draw();
  });
  // Draw score from one bird
  // The generation and
  // Write the highest score
  fill(255);
  rect(0, height - 30, width, height);
  fill(0);
  textSize(20);
  text('Current: ' + birds[0].score.toString().padStart(10, ' ') + ', Generation: ' + generation + ', Highest: ' + highest, 20, height - 10);
}

function generateBirds(generationOne) {
  for (var i = 0; i < TOTAL_BIRDS; i++) {
    birds.push(generationOne ? new Bird() : pickOneBird());
  }
}

function checkGenerationEnd() {
  if (birds.length === 0) {
    calculateFitness();
    generateBirds();
    oldGenerationBirds = [];
    pipes = [];
    counter = 0;
    generation++;
  }
}

function pickOneBird() {
  var index = 0;
  var r = random(1);

  while (r > 0) {
    r -= oldGenerationBirds[index].fitness;
    index++;
  }

  index--;

  // Create new bird
  var newBird = new Bird();

  // Select an old bird and mutate it's brain
  var pickedBird = oldGenerationBirds[index];
  pickedBird.brain.mutate(mutate);

  // Copy the brain to the new bird
  newBird.brain = pickedBird.brain.copy();

  // Return the new bird
  return newBird;
}

function calculateFitness() {
  var sum = oldGenerationBirds.reduce(function (total, bird) {return total += bird.score;}, 0);

  oldGenerationBirds.forEach(function (bird) {
    bird.fitness = bird.score / sum;
  });
}

function mutate(val) {
  if (random(1) < 0.1) {
    var offset = randomGaussian() * 0.5;
    return val + offset;;
  } else {
    return val;
  }
}var

Pipe = function () {
  function Pipe() {_classCallCheck(this, Pipe);
    this.top = random(height / 5, 3 * height / 4);
    this.gapSize = 90;
    this.bottom = this.top + this.gapSize;
    this.x = width;
    this.w = 30;
    this.speed = -5;
  }_createClass(Pipe, [{ key: 'draw', value: function draw()

    {
      fill('#44bd32');
      noStroke();
      rect(this.x, 0, this.w, this.top);
      rect(this.x, this.bottom, this.w, height);
    } }, { key: 'update', value: function update()

    {
      this.x += this.speed;
    } }, { key: 'isOffscreen', value: function isOffscreen()

    {
      return this.x < -this.w;
    } }]);return Pipe;}();var


Bird = function () {
  function Bird() {_classCallCheck(this, Bird);
    this.x = 60;
    this.y = random(height);
    this.yVelocity = 0;
    this.r = 15;
    this.gravity = 0.8;
    this.flyForce = -12;
    this.highlight = false;
    this.score = 0;
    this.fitness = 0;
    this.brain = new NeuralNetwork(5, 10, 2);
  }_createClass(Bird, [{ key: 'draw', value: function draw()

    {
      // fill(255);
      // ellipse(this.x, this.y, this.r * 2);
      imageMode(CENTER);
      image(img, this.x, this.y, this.r * 2, this.r * 2);
    } }, { key: 'think', value: function think(

    pipes) {var _this = this;
      // Get the closes pipe to the bird
      var currentPipe = pipes.find(function (pipe) {return pipe.x + pipe.w > _this.x;});

      // Calculate the inputs to the NN
      var inputs = [];
      inputs.push(this.y / height);
      inputs.push(this.yVelocity / 10);
      inputs.push(currentPipe.top / height);
      inputs.push(currentPipe.bottom / height);
      inputs.push(currentPipe.x / width);

      // Predict outputs
      var outputs = this.brain.predict(inputs);

      // Fly only if ...
      if (outputs[0] > outputs[1]) {
        this.fly();
      }
    } }, { key: 'update', value: function update()

    {
      this.score++;
      this.yVelocity += this.gravity;
      this.yVelocity *= 0.9;
      this.y += this.yVelocity;

      if (this.y > height) {
        this.y = height;
        this.yVelocity = 0;
      }

      if (this.y < 0) {
        this.y = 0;
        this.yVelocity = 0;
      }
    } }, { key: 'fly', value: function fly()

    {
      this.yVelocity += this.flyForce;
    } }, { key: 'hitsPipe', value: function hitsPipe(

    pipe) {
      return (
        (this.y - this.r < pipe.top || this.y + this.r > pipe.bottom) &&
        this.x + this.r > pipe.x &&
        this.x - this.r < pipe.x + pipe.w);

    } }]);return Bird;}();


// Code by Daniel Shiffman on TheCodingTrain
// Other techniques for learning
var
ActivationFunction =
function ActivationFunction(func, dfunc) {_classCallCheck(this, ActivationFunction);
  this.func = func;
  this.dfunc = dfunc;
};


var sigmoid = new ActivationFunction(
function (x) {return 1 / (1 + Math.exp(-x));},
function (y) {return y * (1 - y);});


var tanh = new ActivationFunction(
function (x) {return Math.tanh(x);},
function (y) {return 1 - y * y;});var



NeuralNetwork = function () {
  // TODO: document what a, b, c are
  function NeuralNetwork(a, b, c) {_classCallCheck(this, NeuralNetwork);
    if (a instanceof NeuralNetwork) {
      this.input_nodes = a.input_nodes;
      this.hidden_nodes = a.hidden_nodes;
      this.output_nodes = a.output_nodes;

      this.weights_ih = a.weights_ih.copy();
      this.weights_ho = a.weights_ho.copy();

      this.bias_h = a.bias_h.copy();
      this.bias_o = a.bias_o.copy();
    } else {
      this.input_nodes = a;
      this.hidden_nodes = b;
      this.output_nodes = c;

      this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
      this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
      this.weights_ih.randomize();
      this.weights_ho.randomize();

      this.bias_h = new Matrix(this.hidden_nodes, 1);
      this.bias_o = new Matrix(this.output_nodes, 1);
      this.bias_h.randomize();
      this.bias_o.randomize();
    }

    // TODO: copy these as well
    this.setLearningRate();
    this.setActivationFunction();


  }_createClass(NeuralNetwork, [{ key: 'predict', value: function predict(

    input_array) {

      // Generating the Hidden Outputs
      var inputs = Matrix.fromArray(input_array);
      var hidden = Matrix.multiply(this.weights_ih, inputs);
      hidden.add(this.bias_h);
      // activation function!
      hidden.map(this.activation_function.func);

      // Generating the output's output!
      var output = Matrix.multiply(this.weights_ho, hidden);
      output.add(this.bias_o);
      output.map(this.activation_function.func);

      // Sending back to the caller!
      return output.toArray();
    } }, { key: 'setLearningRate', value: function setLearningRate()

    {var learning_rate = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0.1;
      this.learning_rate = learning_rate;
    } }, { key: 'setActivationFunction', value: function setActivationFunction()

    {var func = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : sigmoid;
      this.activation_function = func;
    } }, { key: 'train', value: function train(

    input_array, target_array) {
      // Generating the Hidden Outputs
      var inputs = Matrix.fromArray(input_array);
      var hidden = Matrix.multiply(this.weights_ih, inputs);
      hidden.add(this.bias_h);
      // activation function!
      hidden.map(this.activation_function.func);

      // Generating the output's output!
      var outputs = Matrix.multiply(this.weights_ho, hidden);
      outputs.add(this.bias_o);
      outputs.map(this.activation_function.func);

      // Convert array to matrix object
      var targets = Matrix.fromArray(target_array);

      // Calculate the error
      // ERROR = TARGETS - OUTPUTS
      var output_errors = Matrix.subtract(targets, outputs);

      // let gradient = outputs * (1 - outputs);
      // Calculate gradient
      var gradients = Matrix.map(outputs, this.activation_function.dfunc);
      gradients.multiply(output_errors);
      gradients.multiply(this.learning_rate);

      // Calculate deltas
      var hidden_T = Matrix.transpose(hidden);
      var weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

      // Adjust the weights by deltas
      this.weights_ho.add(weight_ho_deltas);
      // Adjust the bias by its deltas (which is just the gradients)
      this.bias_o.add(gradients);

      // Calculate the hidden layer errors
      var who_t = Matrix.transpose(this.weights_ho);
      var hidden_errors = Matrix.multiply(who_t, output_errors);

      // Calculate hidden gradient
      var hidden_gradient = Matrix.map(hidden, this.activation_function.dfunc);
      hidden_gradient.multiply(hidden_errors);
      hidden_gradient.multiply(this.learning_rate);

      // Calcuate input->hidden deltas
      var inputs_T = Matrix.transpose(inputs);
      var weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

      this.weights_ih.add(weight_ih_deltas);
      // Adjust the bias by its deltas (which is just the gradients)
      this.bias_h.add(hidden_gradient);
    } }, { key: 'serialize', value: function serialize()

    {
      return JSON.stringify(this);
    } }, { key: 'copy',














    // Adding function for neuro-evolution
    value: function copy() {
      return new NeuralNetwork(this);
    }

    // Accept an arbitrary function for mutation
  }, { key: 'mutate', value: function mutate(func) {
      this.weights_ih.map(func);
      this.weights_ho.map(func);
      this.bias_h.map(func);
      this.bias_o.map(func);
    } }], [{ key: 'deserialize', value: function deserialize(data) {if (typeof data == 'string') {data = JSON.parse(data);}var nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes, data.output_nodes);nn.weights_ih = Matrix.deserialize(data.weights_ih);nn.weights_ho = Matrix.deserialize(data.weights_ho);nn.bias_h = Matrix.deserialize(data.bias_h);nn.bias_o = Matrix.deserialize(data.bias_o);nn.learning_rate = data.learning_rate;return nn;} }]);return NeuralNetwork;}();


// let m = new Matrix(3,2);
var

Matrix = function () {
  function Matrix(rows, cols) {var _this2 = this;_classCallCheck(this, Matrix);
    this.rows = rows;
    this.cols = cols;
    this.data = Array(this.rows).fill().map(function () {return Array(_this2.cols).fill(0);});
  }_createClass(Matrix, [{ key: 'copy', value: function copy()

    {
      var m = new Matrix(this.rows, this.cols);
      for (var i = 0; i < this.rows; i++) {
        for (var j = 0; j < this.cols; j++) {
          m.data[i][j] = this.data[i][j];
        }
      }
      return m;
    } }, { key: 'toArray', value: function toArray()
















    {
      var arr = [];
      for (var i = 0; i < this.rows; i++) {
        for (var j = 0; j < this.cols; j++) {
          arr.push(this.data[i][j]);
        }
      }
      return arr;
    } }, { key: 'randomize', value: function randomize()

    {
      return this.map(function (e) {return Math.random() * 2 - 1;});
    } }, { key: 'add', value: function add(

    n) {
      if (n instanceof Matrix) {
        if (this.rows !== n.rows || this.cols !== n.cols) {
          console.log('Columns and Rows of A must match Columns and Rows of B.');
          return;
        }
        return this.map(function (e, i, j) {return e + n.data[i][j];});
      } else {
        return this.map(function (e) {return e + n;});
      }
    } }, { key: 'multiply', value: function multiply(
























    n) {
      if (n instanceof Matrix) {
        if (this.rows !== n.rows || this.cols !== n.cols) {
          console.log('Columns and Rows of A must match Columns and Rows of B.');
          return;
        }

        // hadamard product
        return this.map(function (e, i, j) {return e * n.data[i][j];});
      } else {
        // Scalar product
        return this.map(function (e) {return e * n;});
      }
    } }, { key: 'map', value: function map(

    func) {
      // Apply a function to every element of matrix
      for (var i = 0; i < this.rows; i++) {
        for (var j = 0; j < this.cols; j++) {
          var val = this.data[i][j];
          this.data[i][j] = func(val, i, j);
        }
      }
      return this;
    } }, { key: 'print', value: function print()







    {
      console.table(this.data);
      return this;
    } }, { key: 'serialize', value: function serialize()

    {
      return JSON.stringify(this);
    } }], [{ key: 'fromArray', value: function fromArray(arr) {return new Matrix(arr.length, 1).map(function (e, i) {return arr[i];});} }, { key: 'subtract', value: function subtract(a, b) {if (a.rows !== b.rows || a.cols !== b.cols) {console.log('Columns and Rows of A must match Columns and Rows of B.');return;} // Return a new Matrix a-b
      return new Matrix(a.rows, a.cols).map(function (_, i, j) {return a.data[i][j] - b.data[i][j];});} }, { key: 'transpose', value: function transpose(matrix) {return new Matrix(matrix.cols, matrix.rows).map(function (_, i, j) {return matrix.data[j][i];});} }, { key: 'multiply', value: function multiply(a, b) {// Matrix product
      if (a.cols !== b.rows) {console.log('Columns of A must match rows of B.');return;}return new Matrix(a.rows, b.cols).map(function (e, i, j) {// Dot product of values in col
        var sum = 0;for (var k = 0; k < a.cols; k++) {sum += a.data[i][k] * b.data[k][j];}return sum;});} }, { key: 'map', value: function map(matrix, func) {// Apply a function to every element of matrix
      return new Matrix(matrix.rows, matrix.cols).map(function (e, i, j) {return func(matrix.data[i][j], i, j);});} }, { key: 'deserialize', value: function deserialize(data) {if (typeof data == 'string') {data = JSON.parse(data);
      }
      var matrix = new Matrix(data.rows, data.cols);
      matrix.data = data.data;
      return matrix;
    } }]);return Matrix;}();


if (typeof module !== 'undefined') {
  module.exports = Matrix;
}