'use client';
// استخدام next/dynamic لتحميل المكون ديناميكيًا على جانب العميل فقط
import dynamic from 'next/dynamic';

// إنشاء مكون ديناميكي
const DynamicEmotionDetector = dynamic(
  () => import('../components/EmotionDetector.js'),
  {
    ssr: false, // هذا يضمن أن المكون لن يتم عرضه على الخادم (Server-Side Rendering)
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
