import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './App.css';
import { Inference } from './components/Inference';
import { ModelComparison } from './components/ModelComparison';

function App() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('inference');

  useEffect(() => {
    loadModels();
  }, []);

  async function loadModels() {
    try {
      if (typeof window.__TAURI__ !== 'undefined') {
        const response = await invoke('get_available_models');
        if (response.success) {
          setModels(response.models);
        }
      } else {
        console.warn('Not running in Tauri - no models available');
        setModels([]);
      }
    } catch (error) {
      console.error('Failed to load models:', error);
      setModels([]);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="loading-screen">
        <h2>🚀 Loading Models...</h2>
        <p>Please wait</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🧠 Hybrid Sentiment Analysis</h1>
        <p>Deep Learning + Classical ML</p>
      </header>

      <nav className="tabs">
        <button
          className={activeTab === 'inference' ? 'active' : ''}
          onClick={() => setActiveTab('inference')}
        >
          💬 Inference
        </button>
        <button
          className={activeTab === 'comparison' ? 'active' : ''}
          onClick={() => setActiveTab('comparison')}
        >
          📊 Comparison
        </button>
      </nav>

      <main className="content">
        {activeTab === 'inference' && <Inference models={models} />}
        {activeTab === 'comparison' && <ModelComparison />}
      </main>

      <footer className="app-footer">
        Pre-trained Models: IMDB (36 combos) | Twitter (36 combos)
      </footer>
    </div>
  );
}

export default App;