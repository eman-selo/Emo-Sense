'use client';
// Use next/dynamic to dynamically load the component on the client side only
import dynamic from 'next/dynamic';

// Create a dynamic component
const DynamicEmotionDetector = dynamic(
  () => import('../components/EmotionDetector.js'),
  {
    ssr: false, // This ensures that the component will not be rendered on the server (Server-Side Rendering)
    loading: () => <p>Loading Emotion Detector</p>,
  }
);

export default function HomePage() {
  return (
    <div style={{ fontFamily: 'Arial, sans-serif', textAlign: 'center', minHeight: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
      <DynamicEmotionDetector />
    </div>
  );
}
