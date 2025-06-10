// TensorFlow.js model service
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const diseaseConfig = require('../config/diseaseConfig');

class ModelService {
  constructor() {
    this.model = null;
    this.isLoaded = false;
    this.modelPath = diseaseConfig.model.path;
    this.inputShape = diseaseConfig.model.inputShape;
  }

  // Load model from file
  async loadModel() {
    try {
      console.log('Loading TensorFlow.js model from Hugging Face...');
  
      const modelUrl = 'https://huggingface.co/reizkafathia/mantoma-tfjs/resolve/main/model.json';
  
      this.model = await tf.loadGraphModel(modelUrl);
      this.isLoaded = true;
  
      console.log('Model loaded');
      console.log(`Input shape: [${this.inputShape.join(', ')}]`);
      console.log(`Classes: ${diseaseConfig.classes.length}`);
  
      return true;
    } catch (error) {
      console.error('Failed to load model:', error);
      this.isLoaded = false;
      throw new Error(`Failed to load model: ${error.message}`);
    }
  }
  
  // Check if model is loaded
  isModelLoaded() {
    return this.isLoaded && this.model !== null;
  }

  // Get basic model info
  getModelInfo() {
    if (!this.isModelLoaded()) {
      return {
        loaded: false,
        message: 'Model not loaded'
      };
    }

    return {
      loaded: true,
      inputShape: this.inputShape,
      outputShape: 'N/A (Graph model)',
      totalParams: 'N/A (Graph model)',
      classes: diseaseConfig.classes.length
    };
  }

  // Prepare image for model
  preprocessImage(imageBuffer) {
    try {
      const imageTensor = tf.node.decodeImage(imageBuffer, 3);
      const resized = tf.image.resizeBilinear(imageTensor, [this.inputShape[0], this.inputShape[1]]);
      const normalized = resized.div(255.0);
      const batched = normalized.expandDims(0);

      imageTensor.dispose();
      resized.dispose();
      normalized.dispose();

      return batched;
    } catch (error) {
      console.error('Image preprocess error:', error);
      throw new Error(`Preprocess failed: ${error.message}`);
    }
  }

  // Run prediction on image
  async predict(imageBuffer) {
    try {
      if (!this.isModelLoaded()) {
        throw new Error('Model not loaded. Call loadModel() first.');
      }

      console.log('Preprocessing image...');
      const processedImage = this.preprocessImage(imageBuffer);

      console.log('Running prediction...');
      const prediction = await this.model.executeAsync(processedImage);
      const predictionData = await prediction.data();

      processedImage.dispose();
      prediction.dispose();

      const predictions = Array.from(predictionData);

      console.log('Prediction done');
      return predictions;

    } catch (error) {
      console.error('Prediction error:', error);
      throw new Error(`Prediction failed: ${error.message}`);
    }
  }

  // Dispose model to free memory
  async unloadModel() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isLoaded = false;
      console.log('Model unloaded');
    }
  }

  // Get memory usage details
  getMemoryInfo() {
    return {
      numTensors: tf.memory().numTensors,
      numBytes: tf.memory().numBytes,
      numBytesInGPU: tf.memory().numBytesInGPU || 0
    };
  }
}

// Export singleton
const modelService = new ModelService();

module.exports = modelService;
