# EmoSense: Real-Time Facial Emotion Classifier

EmoSense is a web-based application built with **React.js** and **TensorFlow.js** that performs real-time facial emotion detection using a webcam. It leverages machine learning models to detect faces and classify emotions such as angry, disgust, fear, happy, sad, surprise, and neutral, displaying results with emojis and labels on a canvas overlay. The application is designed to be user-friendly, with a clean interface and intuitive controls for starting and stopping emotion analysis.

This project was developed by students as part of the **IECD React.js AI 2025** course, showcasing their skills in modern web development and machine learning integration.

## Features
- **Real-Time Face Detection**: Uses the BlazeFace model to detect faces in a video stream from the user's webcam.
- **Emotion Classification**: Classifies detected faces into one of seven emotions using a pre-trained TensorFlow.js model.
- **Dynamic Visualization**: Draws bounding boxes around detected faces and displays the corresponding emotion label and emoji on a canvas.
- **Responsive UI**: Built with React.js and styled using Tailwind CSS for a modern, responsive design.
- **Client-Side Processing**: All computations, including model inference, are performed in the browser, ensuring privacy and no server dependency.
- **Model Caching**: Stores the emotion classification model in IndexedDB for faster subsequent loads.
- **User Controls**: Simple buttons to start and stop the analysis, with clear feedback during model loading and camera access.

## How It Works
EmoSense operates in the following steps:

1. **Model Loading**:
   - The application loads two machine learning models:
     - **BlazeFace**: A lightweight face detection model to identify faces in the webcam feed.
     - **Emotion Classifier**: A custom-trained model to classify emotions based on facial expressions.
   - Models are loaded from a local path (`/model/model.json` for the emotion classifier and `./models/blazeface/model.json` for BlazeFace) or cached in IndexedDB for faster access.

2. **Webcam Access**:
   - Upon clicking "Start Analysis," the application requests access to the user's webcam using the browser's `navigator.mediaDevices.getUserMedia` API.
   - The video stream is displayed in a `<video>` element, with a canvas overlay for real-time annotations.

3. **Face Detection**:
   - The BlazeFace model processes the video feed every 250ms to detect faces, returning their bounding box coordinates (top-left and bottom-right).
   - Detected faces are adjusted to square regions to ensure consistent input for the emotion classifier.

4. **Emotion Classification**:
   - Each detected face is cropped, resized to 48x48 pixels, converted to grayscale, and normalized.
   - The processed face image is passed to the emotion classifier, which predicts one of seven emotions (angry, disgust, fear, happy, sad, surprise, neutral).
   - The prediction is mapped to a label and emoji for display.

5. **Visualization**:
   - Bounding boxes are drawn around detected faces on the canvas, with the corresponding emotion label and emoji displayed nearby.
   - A separate section below the video feed shows snapshots of detected faces with their classified emotions.

6. **Stopping Analysis**:
   - Clicking "Stop Analysis" terminates the webcam stream, clears the canvas, and stops the detection loop.

## Prerequisites
To run EmoSense locally, ensure you have the following:
- **Node.js** (v16 or higher) and **npm** for managing dependencies.
- A modern web browser (Chrome, Firefox, or Edge) with webcam support.
- A webcam connected to your device.
- The pre-trained model files (`model.json` for the emotion classifier and BlazeFace model files) placed in the appropriate directories (`/model/` and `./models/blazeface/`).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd emo-sense
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

3. **Ensure Model Files**:
   - Place the emotion classification model files (`model.json` and associated weights) in the `/public/model/` directory.
   - Place the BlazeFace model files in the `/public/models/blazeface/` directory.
   - Alternatively, update the model paths in `components/EmotionDetector.js` if using a different directory structure.

4. **Run the Application**:
   ```bash
   npm run dev
   ```
   This starts the development server, typically at `http://localhost:3000`.

5. **Access the Application**:
   - Open your browser and navigate to `http://localhost:3000`.
   - Allow webcam access when prompted.

## Usage
1. **Start the Application**:
   - Upon loading, the application displays a "Loading machine learning models" message until the BlazeFace and emotion classifier models are ready.
   - Once loaded, the webcam feed appears in the center of the screen.

2. **Begin Analysis**:
   - Click the **Start Analysis** button to activate the webcam and begin face detection and emotion classification.
   - Faces in the webcam feed will be outlined with red bounding boxes, and the detected emotion (e.g., 😊 happy) will be displayed next to each face.
   - Snapshots of detected faces with their emotions are shown below the video feed.

3. **Stop Analysis**:
   - Click the **Stop Analysis** button to halt the webcam stream and clear the canvas and detected faces section.

4. **Troubleshooting**:
   - If the camera fails to start, ensure you have granted webcam permissions in your browser.
   - If models fail to load, verify that the model files are correctly placed in the `/public/` directory and that the paths in `EmotionDetector.js` are accurate.
   - Check the browser console for detailed error messages.

## Project Structure
```
emo-sense/
├── app/
│   └── page.js              # Main page component with dynamic EmotionDetector import
├── components/
│   └── EmotionDetector.js   # Core component for face detection and emotion classification
├── public/
│   ├── model/               # Directory for emotion classifier model files
│   └── models/blazeface/    # Directory for BlazeFace model files
├── postcss.config.mjs       # PostCSS configuration for Tailwind CSS
├── package.json             # Project dependencies and scripts
└── README.md                # This file
```

## Technologies Used
- **React.js**: For building the user interface with dynamic components.
- **Next.js**: For server-side rendering optimization and dynamic imports.
- **TensorFlow.js**: For client-side machine learning, including model loading and inference.
- **BlazeFace**: A pre-trained model for face detection.
- **Tailwind CSS**: For responsive and modern styling.
- **IndexedDB**: For caching the emotion classifier model to reduce load times.

## Limitations
- **Performance**: Real-time processing may be resource-intensive on low-end devices, as it relies on CPU-based TensorFlow.js (`cpu` backend).
- **Lighting and Camera Quality**: Poor lighting or low-resolution webcams may affect face detection and emotion classification accuracy.
- **Model Accuracy**: The emotion classifier's performance depends on the quality of the training data and may misclassify emotions in certain cases.
- **Browser Compatibility**: Requires a modern browser with WebGL and MediaDevices API support.

## Future Improvements
- Optimize performance by exploring WebGL backend for TensorFlow.js.
- Enhance the emotion classifier with a larger and more diverse training dataset.
- Add support for multiple languages in the UI.
- Implement error handling for edge cases, such as no faces detected for extended periods.
- Add options to adjust detection intervals or model parameters via the UI.

## Contributors
This project was developed by the following students of the **IECD React.js AI 2025** course:
- [Eman Selo](https://www.linkedin.com/in/eman-selo-b8972a36b)
- [Vian Jamal](https://www.linkedin.com/in/vian-jamal-654a8236b)


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Developed during the IECD ReactJS + AI 2025 course.
- Powered by TensorFlow.js, Next.js, and open-source communities.