const regexCell = /cell_[0-9]|lstm_[0-9]/gi;
const regexWeights = /weights|weight|kernel|kernels|w/gi;
const regexFullyConnected = /softmax/gi;

const MANIFEST_FILE = 'manifest.json';

const sampleFromDistribution = (input) => {
    const randomValue = Math.random();
    let sum = 0;
    let result;
    for (let j = 0; j < input.length; j += 1) {
      sum += input[j];
      if (randomValue < sum) {
        result = j;
        break;
      }
    }
    return result;
  };

class CharRNN {

  constructor(modelPath, callback) {
    this.ready = false;
    this.model = {};
    this.cellsAmount = 0;
    this.cells = [];
    this.zeroState = { c: [], h: [] };
    this.state = { c: [], h: [] };
    this.vocab = {};
    this.vocabSize = 0;
    this.probabilities = [];
    this.defaults = {
      seed: 'a',
      length: 20,
      temperature: 0.5,
      stateful: false,
    };
    this.ready = callCallback(this.loadCheckpoints(modelPath), callback);
  }

  resetState() {
    this.state = this.zeroState;
  }

  setState(state) {
    this.state = state;
  }

  getState() {
    return this.state;
  }

  async loadCheckpoints(path) {
    const reader = new CheckpointLoader(path);
    const vars = await reader.getAllVariables();
    Object.keys(vars).forEach((key) => {
      if (key.match(regexCell)) {
        if (key.match(regexWeights)) {
          this.model[`Kernel_${key.match(/[0-9]/)[0]}`] = vars[key];
          this.cellsAmount += 1;
        } else {
          this.model[`Bias_${key.match(/[0-9]/)[0]}`] = vars[key];
        }
      } else if (key.match(regexFullyConnected)) {
        if (key.match(regexWeights)) {
          this.model.fullyConnectedWeights = vars[key];
        } else {
          this.model.fullyConnectedBiases = vars[key];
        }
      } else {
        this.model[key] = vars[key];
      }
    });
    await this.loadVocab(path);
    await this.initCells();
    return this;
  }

  async loadVocab(path) {
    const json = await fetch(`${path}/vocab.json`)
      .then(response => response.json())
      .catch(err => console.error(err));
    this.vocab = json;
    this.vocabSize = Object.keys(json).length;
  }

  async initCells() {
    this.cells = [];
    this.zeroState = { c: [], h: [] };
    const forgetBias = tf.tensor(1.0);

    const lstm = (i) => {
      const cell = (DATA, C, H) =>
        tf.basicLSTMCell(forgetBias, this.model[`Kernel_${i}`], this.model[`Bias_${i}`], DATA, C, H);
      return cell;
    };

    for (let i = 0; i < this.cellsAmount; i += 1) {
      this.zeroState.c.push(tf.zeros([1, this.model[`Bias_${i}`].shape[0] / 4]));
      this.zeroState.h.push(tf.zeros([1, this.model[`Bias_${i}`].shape[0] / 4]));
      this.cells.push(lstm(i));
    }

    this.state = this.zeroState;
  }

  async generateInternal(options) {
    await this.ready;
    const seed = options.seed || this.defaults.seed;
    const length = +options.length || this.defaults.length;
    const temperature = +options.temperature || this.defaults.temperature;
    const stateful = options.stateful || this.defaults.stateful;
    if (!stateful) {
      this.state = this.zeroState;
    }

    const results = [];
    const userInput = Array.from(seed);
    const encodedInput = [];

    userInput.forEach((char) => {
      encodedInput.push(this.vocab[char]);
    });

    let input = encodedInput[0];
    let probabilitiesNormalized = [];

    for (let i = 0; i < userInput.length + length + -1; i += 1) {
      const onehotBuffer = await tf.buffer([1, this.vocabSize]);
      onehotBuffer.set(1.0, 0, input);
      const onehot = onehotBuffer.toTensor();
      let output;
      if (this.model.embedding) {
        const embedded = tf.matMul(onehot, this.model.embedding);
        output = tf.multiRNNCell(this.cells, embedded, this.state.c, this.state.h);
      } else {
        output = tf.multiRNNCell(this.cells, onehot, this.state.c, this.state.h);
      }

      this.state.c = output[0];
      this.state.h = output[1];

      const outputH = this.state.h[1];
      const weightedResult = tf.matMul(outputH, this.model.fullyConnectedWeights);
      const logits = tf.add(weightedResult, this.model.fullyConnectedBiases);
      const divided = tf.div(logits, tf.tensor(temperature));
      const probabilities = tf.exp(divided);
      probabilitiesNormalized = await tf.div(
        probabilities,
        tf.sum(probabilities),
      ).data();

      if (i < userInput.length - 1) {
        input = encodedInput[i + 1];
      } else {
        input = sampleFromDistribution(probabilitiesNormalized);
        results.push(input);
      }
    }

    let generated = '';
    results.forEach((char) => {
      const mapped = Object.keys(this.vocab).find(key => this.vocab[key] === char);
      if (mapped) {
        generated += mapped;
      }
    });
    this.probabilities = probabilitiesNormalized;
    return {
      sample: generated,
      state: this.state,
    };
  }
  reset() {
    this.state = this.zeroState;
  }
  async generate(options, callback) {
    this.reset();
    return callCallback(this.generateInternal(options), callback);
  }
  async predict(temp, callback) {
    let probabilitiesNormalized = [];
    const temperature = temp > 0 ? temp : 0.1;
    const outputH = this.state.h[1];
    const weightedResult = tf.matMul(outputH, this.model.fullyConnectedWeights);
    const logits = tf.add(weightedResult, this.model.fullyConnectedBiases);
    const divided = tf.div(logits, tf.tensor(temperature));
    const probabilities = tf.exp(divided);
    probabilitiesNormalized = await tf.div(
      probabilities,
      tf.sum(probabilities),
    ).data();

    const sample = sampleFromDistribution(probabilitiesNormalized);
    const result = Object.keys(this.vocab).find(key => this.vocab[key] === sample);
    this.probabilities = probabilitiesNormalized;
    if (callback) {
      callback(result);
    }
    /* eslint max-len: ["error", { "code": 180 }] */
    const pm = Object.keys(this.vocab).map(c => ({ char: c, probability: this.probabilities[this.vocab[c]] }));
    return {
      sample: result,
      probabilities: pm,
    };
  }
  async feed(inputSeed, callback) {
    await this.ready;
    const seed = Array.from(inputSeed);
    const encodedInput = [];

    seed.forEach((char) => {
      encodedInput.push(this.vocab[char]);
    });

    let input = encodedInput[0];
    for (let i = 0; i < seed.length; i += 1) {
      const onehotBuffer = await tf.buffer([1, this.vocabSize]);
      onehotBuffer.set(1.0, 0, input);
      const onehot = onehotBuffer.toTensor();
      let output;
      if (this.model.embedding) {
        const embedded = tf.matMul(onehot, this.model.embedding);
        output = tf.multiRNNCell(this.cells, embedded, this.state.c, this.state.h);
      } else {
        output = tf.multiRNNCell(this.cells, onehot, this.state.c, this.state.h);
      }
      this.state.c = output[0];
      this.state.h = output[1];
      input = encodedInput[i];
    }
    if (callback) {
      callback();
    }
  }
}

const charRNN = (modelPath = './', callback) => new CharRNN(modelPath, callback);

export default charRNN;

function callCallback(promise, callback) {
    if (callback) {
      promise
        .then((result) => {
          callback(undefined, result);
          return result;
        })
        .catch((error) => {
          callback(error);
          return error;
        });
    }
    return promise;
  }


export default class CheckpointLoader {
  constructor(urlPath) {
    this.urlPath = urlPath;
    if (this.urlPath.charAt(this.urlPath.length - 1) !== '/') {
      this.urlPath += '/';
    }
  }

  async loadManifest() {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', this.urlPath + "manifest.json");

      xhr.onload = () => {
        this.checkpointManifest = JSON.parse(xhr.responseText);
        resolve();
      };
      xhr.onerror = (error) => {
        reject();
        throw new Error(`${MANIFEST_FILE} not found at ${this.urlPath}. ${error}`);
      };
      xhr.send();
    });
  }

  async getCheckpointManifest() {
    if (this.checkpointManifest == null) {
      await this.loadManifest();
    }
    return this.checkpointManifest;
  }

  async getAllVariables() {
    if (this.variables != null) {
      return Promise.resolve(this.variables);
    }
    await this.getCheckpointManifest();
    const variableNames = Object.keys(this.checkpointManifest);
    const variablePromises = variableNames.map(v => this.getVariable(v));
    return Promise.all(variablePromises).then((variables) => {
      this.variables = {};
      for (let i = 0; i < variables.length; i += 1) {
        this.variables[variableNames[i]] = variables[i];
      }
      return this.variables;
    });
  }
  getVariable(varName) {
    if (!(varName in this.checkpointManifest)) {
      throw new Error(`Cannot load non-existent variable ${varName}`);
    }
    const variableRequestPromiseMethod = (resolve) => {
      const xhr = new XMLHttpRequest();
      xhr.responseType = 'arraybuffer';
      const fname = this.checkpointManifest[varName].filename;
      xhr.open('GET', this.urlPath + fname);
      xhr.onload = () => {
        if (xhr.status === 404) {
          throw new Error(`Not found variable ${varName}`);
        }
        const values = new Float32Array(xhr.response);
        const tensor = tf.Tensor.make(this.checkpointManifest[varName].shape, { values });
        resolve(tensor);
      };
      xhr.onerror = (error) => {
        throw new Error(`Could not fetch variable ${varName}: ${error}`);
      };
      xhr.send();
    };
    if (this.checkpointManifest == null) {
      return new Promise((resolve) => {
        this.loadManifest().then(() => {
          new Promise(variableRequestPromiseMethod).then(resolve);
        });
      });
    }
    return new Promise(variableRequestPromiseMethod);
  }
}