"use client";
import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { loadLayersModel } from '@tensorflow/tfjs-layers';
import '@tensorflow/tfjs-backend-cpu';
import * as blazeface from '@tensorflow-models/blazeface';

//Project constants
const EMOTION_CLASSIFICATION_MODEL_URL = '/model/model.json';
const TF_BACKEND = 'cpu'; // 
const CLASSIFIER_IMG_DIMENSIONS = [48, 48]; //Image dimensions required for emtion classification
const PREDICTION_INTERVAL_MS = 250; //Time interval between each detection in ms

// تعريف المشاعر والرموز التعبيرية المقابلة
const EMOTION = {
    0: { label: 'angry', emoji: '😠' },
    1: { label: 'disgust', emoji: '🤢' },
    2: { label: 'fear', emoji: '😨' },
    3: { label: 'happy', emoji: '🙂' },
    4: { label: 'sad', emoji: '🙁' },
    5: { label: 'surprise', emoji: '😯' },
    6: { label: 'neutral', emoji: '😐' },
};

// متغيرات لتخزين النماذج المحملة لتجنب إعادة التحميل المتكرر
let emotionClassifierModel = null;
let blazefaceModel = null;
const EMOTION_MODEL_STORAGE_NAME = 'emotion-classifier-model';


const loadTfModels = async () => {

    if (!emotionClassifierModel) {
        console.log('Trying to load EmotionClassifier model from IndexedDB');
        try {
            const modelsInDB = await tf.io.listModels();
            if (modelsInDB[`indexeddb://${EMOTION_MODEL_STORAGE_NAME}`]) {
                emotionClassifierModel = await loadLayersModel(`indexeddb://${EMOTION_MODEL_STORAGE_NAME}`);
                console.log('EmotionClassifier has been loaded from IndexedDB');
            } else {
                console.log('EmotionClassifier model not found in IndexedDB, loading from network...');
                emotionClassifierModel = await loadLayersModel(EMOTION_CLASSIFICATION_MODEL_URL);
                console.log('EmotionClassifier has been loaded from network');
                await emotionClassifierModel.save(`indexeddb://${EMOTION_MODEL_STORAGE_NAME}`);
                console.log('The EmotionClassifier model has been saved in IndexedDB');
            }
        } catch (error) {
            console.error("Failed to load or save the EmotionClassifier model", error);
            throw new Error(`Failed to load EmotionClassifier model: ${error.message}`);
        }
    }


    if (!blazefaceModel) {
        console.log('loading blazeface model');
        try {

            const localBlazefaceModelPath = './models/blazeface/model.json';
            blazefaceModel = await blazeface.load({ modelUrl: localBlazefaceModelPath });
            console.log('blazeface model has been loaded');
        } catch (error) {
            console.error("failed to load blazeface model", error);
            throw new Error("blazeface model loading failed. Please check the model path.");
        }
    }
};

/**
 * يكتشف الوجوه في عنصر الفيديو ويعيد مواقعها مربعة.
 * @param {HTMLVideoElement} videoElement عنصر الفيديو الذي يتم تحليله.
 * @returns {Array<Object>} مصفوفة من كائنات الوجه المكتشفة، كل منها يحتوي على topLeft و bottomRight.
 */
const getFacesRects = async (videoElement) => {
    if (!blazefaceModel) {
        console.error('The blazeface model has not been loaded yet');
        return [];
    }

    const returnTensors = false;
    const flipHorizontal = false;
    const annotateBoxes = false;

    const facePositions = await blazefaceModel.estimateFaces(videoElement, returnTensors, flipHorizontal, annotateBoxes);
    for (const position of facePositions) {
        const x1 = position.topLeft[0];
        const y1 = position.topLeft[1];
        const x2 = position.bottomRight[0];
        const y2 = position.bottomRight[1];

        const width = x2 - x1;
        const height = y2 - y1;

        const maxDimension = Math.max(width, height);

        const dx = (maxDimension - width) / 2;
        const dy = (maxDimension - height) / 2;

        position.topLeft[0] = x1 - dx;
        position.topLeft[1] = y1 - dy;
        position.bottomRight[0] = x2 + dx;
        position.bottomRight[1] = y2 + dy;

        position.topLeft[0] = Math.max(0, position.topLeft[0]);
        position.topLeft[1] = Math.max(0, position.topLeft[1]);
        position.bottomRight[0] = Math.min(videoElement.videoWidth, position.bottomRight[0]);
        position.bottomRight[1] = Math.min(videoElement.videoHeight, position.bottomRight[1]);
    }
    return facePositions;
};

/**
 * يقص صورة الوجه من التنسور الأصلي ويعالجها لتكون جاهزة لنموذج تصنيف المشاعر.
 * @param {tf.Tensor} imgTensor التنسور الكامل للصورة الأصلية.
 * @param {Object} position كائن يحتوي على topLeft و bottomRight للوجه المكتشف.
 * @returns {tf.Tensor} تنسور صورة الوجه المعالج (48x48، تدرج رمادي، مطبع).
 */
const getFaceImage = async (imgTensor, position) => {
    const imgDimensions = tf.tensor2d([imgTensor.shape[1], imgTensor.shape[2]], [1, 2]);

    const normalizedTopLeft = tf.div(tf.tensor1d([position.topLeft[1], position.topLeft[0]]), imgDimensions).dataSync();
    const normalizedBottomRight = tf.div(tf.tensor1d([position.bottomRight[1], position.bottomRight[0]]), imgDimensions).dataSync();

    const box = tf.concat([normalizedTopLeft, normalizedBottomRight]).expandDims(0);
    const faceImage = tf.image.cropAndResize(imgTensor, box, [0], CLASSIFIER_IMG_DIMENSIONS);
    const normalizedFaceImage = faceImage.div(tf.scalar(255));

    return normalizedFaceImage
        .mean(3, false)
        .toFloat()
        .expandDims(3);
};

/**
 * Classifies the emotion of a face
 * @param {tf.Tensor} img - face only; size 48x48; grayscale;
 * @returns {Object} - emotion containing 'label' and 'emoji'
 */
const classifyEmotion = async (img) => {
    if (!emotionClassifierModel) {
        console.error('EmotionClassifier model has not been loaded yet');
        return { label: 'unknown', emoji: '❓' };
    }
    const predictions = await emotionClassifierModel.predict(img).data();
    const indexOfMaxValue = predictions.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
    return EMOTION[indexOfMaxValue];
};

/**
 * يرسم مربعات الوجوه والمشاعر على عنصر الـ Canvas.
 * @param {CanvasRenderingContext2D} canvasCtx سياق رسم الـ Canvas.
 * @param {HTMLVideoElement} videoElement عنصر الفيديو لضبط الأبعاد.
 * @param {Array<Object>} facePositions مصفوفة مواقع الوجوه.
 * @param {Array<Object>} emotions مصفوفة المشاعر المقابلة لكل وجه.
 */
const drawResults = (canvasCtx, videoElement, facePositions, emotions) => {
    canvasCtx.clearRect(0, 0, videoElement.videoWidth, videoElement.videoHeight);

    facePositions.forEach((position, index) => {
        const { topLeft, bottomRight } = position;
        const emotion = emotions[index];

        const x = topLeft[0];
        const y = topLeft[1];
        const width = bottomRight[0] - topLeft[0];
        const height = bottomRight[1] - topLeft[1];

        canvasCtx.strokeStyle = 'red';
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeRect(x, y, width, height);

        canvasCtx.fillStyle = 'red';
        canvasCtx.font = '20px Arial';
        canvasCtx.fillText(`${emotion.emoji} ${emotion.label}`, x, y > 20 ? y - 10 : y + height + 20);
    });
};

// -----------------------------------------------------
// مكون React الرئيسي EmotionDetector
// -----------------------------------------------------

const EmotionDetector = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [isDetecting, setIsDetecting] = useState(false);
    const [loadingModels, setLoadingModels] = useState(true);
    const [detectedFacesData, setDetectedFacesData] = useState([]); // <-- حالة جديدة لتخزين بيانات الوجوه
    const intervalIdRef = useRef(null);

    // useEffect لتحميل النماذج عند تحميل المكون لأول مرة
    useEffect(() => {
        const setupTf = async () => {
            try {
                await tf.setBackend(TF_BACKEND);
                await loadTfModels();
            } catch (error) {
                console.error("Failed to load Tensorflow.js models", error);
                alert(`Failed to load machine learning models : ${error.message}`);
            } finally {
                setLoadingModels(false);
            }
        };
        setupTf();
    }, []);

    // دالة لبدء عملية اكتشاف الوجه والمشاعر
    const startDetection = useCallback(async () => {
        if (loadingModels) {
            console.log("Models are still loading , please wait..");
            return;
        }
        if (isDetecting) {
            console.log("The analysis is already running.");
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoRef.current.srcObject = stream;

            await new Promise((resolve) => {
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current.play();
                    resolve();
                };
            });

            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
            const canvasCtx = canvasRef.current.getContext('2d');

            setIsDetecting(true);
            console.log(`Start analysis every ${PREDICTION_INTERVAL_MS} ms`);

            intervalIdRef.current = setInterval(async () => {
                if (videoRef.current.readyState < 2 || !blazefaceModel || !emotionClassifierModel) {
                    return;
                }

                const imgTensor = tf.browser.fromPixels(videoRef.current).expandDims(0);
                let currentDetectedFaces = [];

                try {
                    const facePositions = await getFacesRects(videoRef.current);
                    if (facePositions.length > 0) {
                        const faceProcessingPromises = facePositions.map(async (position) => {
                            const faceImageTensor = await getFaceImage(imgTensor, position);
                            const emotion = await classifyEmotion(faceImageTensor);

                            // --- الجزء المعدل: استخدام tf.browser.draw لزيادة الكفاءة ---
                            const tempCanvas = document.createElement('canvas');
                            tempCanvas.width = CLASSIFIER_IMG_DIMENSIONS[0];
                            tempCanvas.height = CLASSIFIER_IMG_DIMENSIONS[1];

                            // tf.browser.draw يتوقع تنسورًا ذو 3 قنوات (RGB) أو 4 قنوات (RGBA).
                            // faceImageTensor لديك هو تدرج رمادي (قناة واحدة).
                            // لذا، نقوم بتكرار القناة الواحدة إلى 3 قنوات للعرض.
                            const faceImgRgb = faceImageTensor.tile([1, 1, 1, 3]);

                            await tf.browser.draw(faceImgRgb.squeeze(), tempCanvas); // استخدام squeeze() لإزالة بُعد الدفعة
                            const imageDataUrl = tempCanvas.toDataURL('image/png');


                            // التخلص من التنسورات المؤقتة لمنع تسرب الذاكرة
                            faceImageTensor.dispose();
                            faceImgRgb.dispose(); // التخلص من التنسور المكرر أيضًا

                            return {
                                position,
                                emotion,
                                imageDataUrl,
                            };
                        });

                        currentDetectedFaces = await Promise.all(faceProcessingPromises);
                    }
                } catch (error) {
                    console.error("Error in the detection or classification process : ", error);
                } finally {
                    imgTensor.dispose(); // التخلص من تنسور إطار الفيديو الأصلي
                }

                setDetectedFacesData(currentDetectedFaces);

                if (currentDetectedFaces.length > 0) {
                    const positions = currentDetectedFaces.map(f => f.position);
                    const emotions = currentDetectedFaces.map(f => f.emotion);
                    drawResults(canvasCtx, videoRef.current, positions, emotions);
                } else {
                    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                }
            }, PREDICTION_INTERVAL_MS);

        } catch (error) {
            console.error("There was an error accessing the camera :", error);
            alert('Camera cannot be accessed, please make sure to allow access.');
            setIsDetecting(false);
        }
    }, [loadingModels, isDetecting]);

    // دالة لإيقاف عملية الاكتشاف
    const stopDetection = useCallback(() => {
        if (videoRef.current && videoRef.current.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        if (intervalIdRef.current) {
            clearInterval(intervalIdRef.current);
            intervalIdRef.current = null;
        }
        if (canvasRef.current) {
            canvasRef.current.getContext('2d').clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
        setIsDetecting(false);
        setDetectedFacesData([]); // مسح الوجوه المكتشفة عند الإيقاف
        console.log('Analysis has been stopped.');
    }, []);

    // useEffect للتنظيف عند إلغاء تحميل المكون
    useEffect(() => {
        return () => {
            stopDetection();
        };
    }, [stopDetection]);

    return (
        <div className="flex flex-col items-center p-8 bg-white rounded-xl shadow-lg max-w-4xl mx-auto my-6">
            <h1 className="text-2xl md:text-3xl lg:text-4xl font-extrabold text-gray-800 mb-4 text-center leading-tight">
                Emotion classifier for faces from camera
            </h1>
            <div className="relative w-full max-w-2xl mx-auto border-2 border-gray-300 rounded-lg overflow-hidden bg-black flex justify-center items-center min-h-96 mb-6"
                style={{ display: loadingModels ? 'flex' : 'block' }}>
                {loadingModels && <p className="text-gray-600 text-lg italic p-4">Loading machine learning models ...Please wait </p>}
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className="w-full h-full object-cover"
                    style={{ display: loadingModels ? 'none' : 'block' }}
                ></video>
                <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full"
                    style={{ display: loadingModels ? 'none' : 'block' }}
                ></canvas>
            </div>
            <div className="mt-8 flex justify-center space-x-4">
                <button
                    onClick={startDetection}
                    disabled={isDetecting || loadingModels}
                    className="px-6 py-3 text-lg font-semibold rounded-lg shadow-md transition-all duration-200
               bg-green-600 hover:bg-green-700 text-white
               disabled:bg-gray-300 disabled:text-gray-500 disabled:cursor-not-allowed
               hover:scale-105 active:scale-100"
                >
                    {loadingModels ? 'Loading...' : 'Start analysis'}
                </button>
                <button
                    onClick={stopDetection}
                    disabled={!isDetecting}
                    className="px-6 py-3 text-lg font-semibold rounded-lg shadow-md transition-all duration-200
               bg-red-600 hover:bg-red-700 text-white
               disabled:bg-gray-300 disabled:text-gray-500 disabled:cursor-not-allowed
               hover:scale-105 active:scale-100"
                >
                    Stop analysis
                </button>
            </div>

            {/* قسم عرض الوجوه المكتشفة */}
            <h2 className="text-2xl font-bold text-gray-700 mt-5 mb-4 text-center">Detected Faces</h2>
            <div className="flex flex-wrap justify-center gap-4 p-4 border border-gray-200 rounded-lg bg-gray-50 max-w-full">
                {detectedFacesData.length > 0 ? (
                    detectedFacesData.map((face, index) => (
                        <div key={index} className="flex flex-col items-center p-3 border border-gray-300 rounded-md shadow-sm bg-white">
                            <img src={face.imageDataUrl} alt={`Detected Face ${index}`} className="w-20 h-20 object-cover rounded-sm mb-2 border border-gray-200" />
                            <p className="text-base font-semibold text-gray-800">{face.emotion.emoji} {face.emotion.label}</p>
                        </div>
                    ))
                ) : (
                    <p className="text-gray-500 italic">No faces detected yet.</p>
                )}
            </div>
        </div>
    );
};

export default EmotionDetector;
