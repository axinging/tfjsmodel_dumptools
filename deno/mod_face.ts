// How to : deno run --allow-read --allow-write --allow-net --unstable
// mod_face.ts
// import './tf-core.es2017.js';
// import './tf-backend-webgpu.es2017.js';
import './tf.js';
import * as jpegts from 'https://deno.land/x/jpegts@1.1/mod.ts';
;
(window as any)
    .document = {}  // polyfill for browser detection
                    // import * as tf from
                    // 'https://cdn.skypack.dev/@tensorflow/tfjs'
                    // console.log(tf); console.log(JSON.stringify(tf));
                tf.env()
                    .set('IS_BROWSER', true);
import {PlatformBrowser} from 'https://cdn.skypack.dev/@tensorflow/tfjs-core/dist/platforms/platform_browser'
// console.log(tf);
console.log(PlatformBrowser);
tf.env().setPlatform('browser', new PlatformBrowser());
import 'https://cdn.skypack.dev/@tensorflow/tfjs-backend-webgpu'
import * as faceDetection from 'https://cdn.skypack.dev/@tensorflow-models/face-detection';
import * as poseDetection from 'https://cdn.skypack.dev/@tensorflow-models/pose-detection';
import {getPredictionData} from './util.js';
import {dump, getIntermediateTensorInfo} from './dump.js';


export function getGraphModel(model) {
  if (model instanceof tf.GraphModel) {
    return model;
  } else if (model.model instanceof tf.GraphModel) {
    return model.model;
  } else if (
      model.baseModel && model.baseModel.model instanceof tf.GraphModel) {
    return model.baseModel.model;
  } else if (model.moveNetModel) {
    // model.moveNetModel instanceof tf.GraphModel
    return model.moveNetModel;
  } else {
    console.warn(`Model doesn't support dump!`);
    return null;
  }
}

async function predictAndGetData(
    model, predict: (model: any, image: any) => Promise<any>, image) {
  let enableDump = true
  const prediction = await predict(model, image);
  let intermediateData = {};
  if (enableDump) {
    const graphModel = getGraphModel(model);
    if (graphModel) {
      intermediateData =
          await getIntermediateTensorInfo(graphModel.getIntermediateTensors());
      graphModel.disposeIntermediateTensors();
    }
  }
  // const predictionData = await getPredictionData(prediction);
  return {data: prediction, intermediateData};
}


tf.env().set('KEEP_INTERMEDIATE_TENSORS', true);

// initialize tensorflow
await tf.setBackend('cpu');
// await tf.setBackend('cpu')
await tf.ready();

console.log('isBrowser:', tf.device_util.isBrowser());
console.log('IS_BROWSER:', tf.env().getBool('IS_BROWSER'));

async function dumpPoseDetection() {
  const detectorConfig = {
    modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
  };
  const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet, detectorConfig)

  const file = await Deno.readFile('./sample.jpg')
  const image = await jpegts.decode(file)

  let predict =
      async (model, image) => {
    return await model.estimatePoses(image);
  }

  const poses = await predictAndGetData(detector, predict, image);
  await detector.estimatePoses(image);
  console.log(poses['intermediateData'])
}

async function dumpFaceDetection() {
  const inputResolution = 128;
  const input = tf.zeros([1, inputResolution, inputResolution, 3]);

  const modelSize = inputResolution === 128 ? 'short' : 'full';
  const url =
      `https://tfhub.dev/mediapipe/tfjs-model/face_detection/${modelSize}/1`;
  const model = await tf.loadGraphModel(url, {fromTFHub: true});

  const enableDump = true;
  let predict = async (model, image) => {
    return await model.predict(image);
  };

  // step 1: run the expected backend.
  const expectedBackend = 'cpu';
  await tf.setBackend(expectedBackend);
  await tf.ready();
  const expectedResult = await predictAndGetData(model, predict, input);
  // console.log(expectedResult['intermediateData']);

  // step 2: run the actual backend.
  const actualBackend = 'webgpu';
  await tf.setBackend(actualBackend);
  await tf.ready();
  const actualResult = await predictAndGetData(model, predict, input);
  // console.log(actualResult['intermediateData']);

  // step3: dump.
  if (enableDump) {
    const dumpLevel = 0;
    const dumpLength = 1;
    const dumpPrefix = 'face_detection';
    const dumpInput = {
      [actualBackend]: actualResult['intermediateData'],
      [expectedBackend + 'expected']: expectedResult['intermediateData'],
    };
    await dump(model, dumpInput, dumpPrefix, dumpLevel, dumpLength);
    console.log('Dump end');
  }
}

await dumpFaceDetection();
