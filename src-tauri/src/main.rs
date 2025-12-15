#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct PredictionResponse {
    success: bool,
    prediction: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct AvailableModelsResponse {
    success: bool,
    models: Vec<ModelInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelInfo {
    dataset: String,
    #[serde(rename = "type")]
    model_type: String,
    encoder: String,
    classifier: Option<String>,
    path: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelResult {
    model: String,
    dataset: String,
    sentiment: String,
    confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ComparisonResponse {
    comparisons: Vec<ModelResult>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CompareRequest {
    text: String,
    dataset: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct AvailableDatasetsResponse {
    success: bool,
    datasets: Vec<String>,
}

#[tauri::command]
async fn predict_sentiment(
    text: String,
    dataset: String,
    model_type: String,
    encoder: String,
    classifier: Option<String>,
) -> Result<PredictionResponse, String> {
    let client = reqwest::Client::new();
    
    let mut body = serde_json::json!({
        "text": text,
        "dataset": dataset,
        "model_type": model_type,
        "encoder": encoder,
    });
    
    if let Some(clf) = classifier {
        body["classifier"] = serde_json::Value::String(clf);
    }
    
    let response = client
        .post("http://127.0.0.1:8000/predict")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Backend error: {}", e))?;
    
    let result: PredictionResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;
    
    Ok(result)
}

#[tauri::command]
async fn get_available_models() -> Result<AvailableModelsResponse, String> {
    let client = reqwest::Client::new();
    
    let response = client
        .get("http://127.0.0.1:8000/models/available")
        .send()
        .await
        .map_err(|e| format!("Backend error: {}", e))?;
    
    let result: AvailableModelsResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;
    
    Ok(result)
}

#[tauri::command]
async fn get_available_datasets() -> Result<AvailableDatasetsResponse, String> {
    let client = reqwest::Client::new();
    
    let response = client
        .get("http://127.0.0.1:8000/datasets/available")
        .send()
        .await
        .map_err(|e| format!("Backend error: {}", e))?;
    
    let result: AvailableDatasetsResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;
    
    Ok(result)
}

#[tauri::command]
async fn compare_models(text: String, dataset: String) -> Result<ComparisonResponse, String> {
    let client = reqwest::Client::new();
    
    let response = client
        .post("http://127.0.0.1:8000/compare")
        .json(&serde_json::json!({
            "text": text,
            "dataset": dataset
        }))
        .send()
        .await
        .map_err(|e| format!("Failed to connect to backend: {}. Make sure Python backend is running on port 8000.", e))?;
    
    if !response.status().is_success() {
        return Err(format!("Backend returned error: {}", response.status()));
    }
    
    let result: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;
    
    if result["success"].as_bool().unwrap_or(false) {
        let comparisons: Vec<ModelResult> = serde_json::from_value(result["comparisons"].clone())
            .map_err(|e| format!("Failed to parse comparisons: {}", e))?;
        Ok(ComparisonResponse { comparisons })
    } else {
        Err(result["error"].as_str().unwrap_or("Unknown error").to_string())
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            predict_sentiment,
            get_available_models,
            get_available_datasets,
            compare_models
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}