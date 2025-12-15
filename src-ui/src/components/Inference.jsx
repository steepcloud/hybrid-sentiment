import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './Inference.css';

export function Inference({ models }) {
  const [text, setText] = useState('');
  const [dataset, setDataset] = useState('imdb');
  const [modelType, setModelType] = useState('hybrid');
  const [encoder, setEncoder] = useState('lstm');
  const [classifier, setClassifier] = useState('xgboost');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const datasets = ['imdb', 'twitter'];
  const encoders = ['lstm', 'gru', 'transformer', 'bert', 'roberta', 'distilbert'];
  const classifiers = ['xgboost', 'random_forest', 'logistic_regression'];

  async function handlePredict() {
    if (!text.trim()) {
      setError('Please enter text');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      console.log('📡 Calling Tauri backend with:', { text, dataset, modelType, encoder, classifier });
      
      const response = await invoke('predict_sentiment', {
        text: text,
        dataset: dataset,
        modelType: modelType,
        encoder: encoder,
        classifier: modelType === 'hybrid' ? classifier : null,
      });

      console.log('✅ Backend response:', response);

      if (response.success) {
        setResult(response.prediction);
      } else {
        setError(response.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('❌ Prediction error:', err);
      setError(`Failed to call backend: ${String(err)}`);
    } finally {
      setLoading(false);
    }
  }

  const examples = {
    positive: "This movie was absolutely fantastic! The acting was superb.",
    negative: "Terrible experience. Would not recommend to anyone.",
  };

  return (
    <div className="inference">
      <div className="sidebar">
        <h3>⚙️ Configuration</h3>
        
        <div className="form-group">
          <label>Dataset:</label>
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map(d => <option key={d} value={d}>{d.toUpperCase()}</option>)}
          </select>
        </div>

        <div className="form-group">
          <label>Model Type:</label>
          <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
            <option value="hybrid">Hybrid</option>
            <option value="end-to-end">End-to-End</option>
          </select>
        </div>

        <div className="form-group">
          <label>Encoder:</label>
          <select value={encoder} onChange={(e) => setEncoder(e.target.value)}>
            {encoders.map(e => <option key={e} value={e}>{e.toUpperCase()}</option>)}
          </select>
        </div>

        {modelType === 'hybrid' && (
          <div className="form-group">
            <label>Classifier:</label>
            <select value={classifier} onChange={(e) => setClassifier(e.target.value)}>
              <option value="xgboost">XGBoost</option>
              <option value="random_forest">Random Forest</option>
              <option value="logistic_regression">Logistic Regression</option>
            </select>
          </div>
        )}

        <div className="examples">
          <p><strong>Examples:</strong></p>
          <button onClick={() => setText(examples.positive)}>Positive</button>
          <button onClick={() => setText(examples.negative)}>Negative</button>
        </div>
      </div>

      <div className="main-panel">
        <h3>💬 Text Input</h3>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to analyze..."
          rows={8}
        />

        <button
          onClick={handlePredict}
          disabled={loading || !text.trim()}
          className="predict-btn"
        >
          {loading ? '⏳ Analyzing...' : '🚀 Predict'}
        </button>

        {error && <div className="error">❌ {error}</div>}

        {result && (
          <div className={`result ${result.sentiment.toLowerCase()}`}>
            <h3>📊 Result</h3>
            <div className="result-row">
              <span>Sentiment:</span>
              <strong>{result.sentiment}</strong>
            </div>
            <div className="result-row">
              <span>Confidence:</span>
              <strong>{(result.confidence * 100).toFixed(1)}%</strong>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}