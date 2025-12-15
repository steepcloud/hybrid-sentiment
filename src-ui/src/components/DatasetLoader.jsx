import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

export function DatasetLoader() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select a file first");
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      const text = await file.text();
      await invoke("load_dataset", { data: text });
      setMessage("Dataset loaded successfully!");
    } catch (error) {
      setMessage(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dataset-loader">
      <h2>Load Dataset</h2>
      <input type="file" accept=".csv,.json" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? "Loading..." : "Upload Dataset"}
      </button>
      {message && <p className={message.includes("Error") ? "error" : "success"}>{message}</p>}
    </div>
  );
}