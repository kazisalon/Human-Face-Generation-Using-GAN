// src/App.jsx
import { useState } from 'react';
import axios from 'axios';
import { Loader2 } from 'lucide-react';
import FaceGenerator from './components/FaceGenerator';
import Header from './components/Header';
import './App.css';

function App() {
  const [generatedImage, setGeneratedImage] = useState(null);
  const [seed, setSeed] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generateFace = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const formData = new FormData();
      if (seed) {
        formData.append('seed', seed);
      }
      
      const response = await axios.post('http://localhost:5000/api/generate', formData, {
        responseType: 'blob'
      });
      
      const imageUrl = URL.createObjectURL(response.data);
      setGeneratedImage(imageUrl);
    } catch (err) {
      console.error('Error generating face:', err);
      setError('Failed to generate face. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSeedChange = (e) => {
    setSeed(e.target.value);
  };

  const downloadImage = () => {
    if (!generatedImage) return;
    
    const a = document.createElement('a');
    a.href = generatedImage;
    a.download = `generated-face-${seed || 'random'}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <FaceGenerator 
            seed={seed}
            onSeedChange={handleSeedChange}
            onGenerate={generateFace}
            loading={loading}
          />
        </div>

        {loading && (
          <div className="text-center py-12">
            <Loader2 className="animate-spin h-12 w-12 mx-auto text-blue-500" />
            <p className="mt-4 text-lg text-gray-600">Generating your unique face...</p>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md mb-6">
            {error}
          </div>
        )}

        {generatedImage && !loading && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">Generated Face</h2>
            <div className="flex flex-col items-center">
              <img 
                src={generatedImage} 
                alt="Generated face" 
                className="max-w-full rounded shadow-md mb-4" 
              />
              <div className="text-sm text-gray-500 mb-4">
                Seed: {seed || 'Random'}
              </div>
              <button
                onClick={downloadImage}
                className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-colors"
              >
                Download Image
              </button>
            </div>
          </div>
        )}
      </main>
      
      <footer className="bg-gray-800 text-white py-6 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p>Human Face Generation using GAN - Built with React, Vite, and Tailwind CSS</p>
        </div>
      </footer>
    </div>
  );
}

export default App;