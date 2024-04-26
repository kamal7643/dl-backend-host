const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());
const upload = multer();

// Load the TensorFlow.js model
let model;
const loadModel = async () => {
  if (!model) {
    model = await tf.loadLayersModel('https://firebasestorage.googleapis.com/v0/b/aplhasecond.appspot.com/o/model.json?alt=media&token=04736c4a-bfbb-40b9-bc73-1368c9185f67');
    console.log('Model loaded successfully');
  }
};


// Preprocess the base64 image
const preprocessBase64Image = async (base64String) => {
  const buffer = Buffer.from(base64String, 'base64');
  const tensorBuffer = tf.node.decodeImage(buffer);
  const resizedTensor = tf.image.resizeBilinear(tensorBuffer, [224, 224]);
  const normalizedTensor = resizedTensor.div(255.0);
  const batchedTensor = normalizedTensor.expandDims(0);
  return batchedTensor;
};

app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.body.image) {
      return res.status(400).json({ error: 'No image provided in the request body' });
    }

    // console.log(req.body);
    console.log("requested !!!")

    if(!model)await loadModel();

    const {image } =  req.body;
    const inputTensor = await preprocessBase64Image(image);
    const output = model.predict(inputTensor);
    const prediction = output.dataSync();

    // Clean up resources
    inputTensor.dispose();
    output.dispose();

    res.json({ prediction});
  } catch (error) {
    console.error('Error processing prediction:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
