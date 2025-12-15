import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./ModelComparison.css";

export function ModelComparison() {
  const [text, setText] = useState("");
  const [dataset, setDataset] = useState("all");
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  // Fetch available datasets on component mount
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        if (typeof window.__TAURI__ !== 'undefined') {
          const response = await invoke("get_available_datasets");
          if (response.success) {
            setAvailableDatasets(response.datasets);
          }
        } else {
          // Mock datasets for development
          setAvailableDatasets(["imdb", "twitter", "custom"]);
        }
      } catch (error) {
        console.error("Failed to fetch datasets:", error);
        setAvailableDatasets(["imdb", "twitter"]); // Fallback
      }
    };

    fetchDatasets();
  }, []);

  const handleCompare = async () => {
    if (!text.trim()) {
      alert("Please enter some text");
      return;
    }

    setLoading(true);
    try {
      if (typeof window.__TAURI__ !== 'undefined') {
        const response = await invoke("compare_models", { 
          text: text,
          dataset: dataset 
        });
        setResults(response.comparisons);
      } else {
        // Mock comparison data
        await new Promise(resolve => setTimeout(resolve, 1500));
        const sentiment = text.toLowerCase().includes('good') ? 'Positive' : 'Negative';
        setResults([
          { model: "LSTM + XGBoost", dataset: "imdb", sentiment, confidence: 0.87 },
          { model: "GRU + Random Forest", dataset: "twitter", sentiment, confidence: 0.82 },
          { model: "Transformer", dataset: "imdb", sentiment, confidence: 0.91 }
        ]);
      }
    } catch (error) {
      alert(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="model-comparison">
      <h2>📊 Compare Models</h2>
      
      <div className="dataset-selector">
        <label>Dataset:</label>
        <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
          <option value="all">ALL</option>
          {availableDatasets.map(ds => (
            <option key={ds} value={ds}>{ds.toUpperCase()}</option>
          ))}
        </select>
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to analyze across all models..."
        rows={4}
      />
      <button onClick={handleCompare} disabled={loading}>
        {loading ? "⏳ Comparing..." : "🔍 Compare All Models"}
      </button>
      
      {results.length > 0 && (
        <div className="results">
          <h3>Results {dataset === 'all' ? 'for All Datasets' : `for ${dataset.toUpperCase()} Dataset`}</h3>
          {results.map((result, idx) => (
            <div key={idx} className={`result-card ${result.sentiment.toLowerCase()}`}>
              <h4>{result.model}</h4>
              <p>Dataset: <strong>{result.dataset.toUpperCase()}</strong></p>
              <p>Sentiment: <strong>{result.sentiment}</strong></p>
              <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}