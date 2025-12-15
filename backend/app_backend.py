from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.inference import HybridSentimentPredictor, EndToEndPredictor

app = FastAPI(title="Hybrid Sentiment Analysis API")

# Enable CORS for Tauri
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "tauri://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store loaded predictors in memory
predictors: Dict[str, Any] = {}

class PredictionRequest(BaseModel):
    text: str
    dataset: str
    model_type: str
    encoder: Optional[str] = 'lstm'
    classifier: Optional[str] = 'xgboost'

class CompareRequest(BaseModel):
    text: str
    dataset: str = 'imdb'  # default dataset

class DatasetStatsRequest(BaseModel):
    dataset: str

@app.get("/")
def root():
    return {"message": "Hybrid Sentiment Analysis API", "status": "running"}

@app.post("/predict")
async def predict_sentiment(request: PredictionRequest):
    print(f"\n🔵 PREDICTION REQUEST from Tauri:")
    print(f"   Text: {request.text}")
    print(f"   Dataset: {request.dataset}")
    print(f"   Model: {request.model_type}")
    print(f"   Encoder: {request.encoder}")
    print(f"   Classifier: {request.classifier}\n")

    try:
        predictor_key = f"{request.dataset}_{request.model_type}_{request.encoder}"
        if request.model_type == 'hybrid':
            predictor_key += f"_{request.classifier}"
        
        if predictor_key not in predictors:
            if request.model_type == 'hybrid':
                model_path = f"results/models/classical_ml/{request.dataset}/{request.encoder}/{request.classifier}.pkl"
                encoder_path = f"results/models/deep_learning/{request.dataset}/{request.encoder}/{request.encoder}_best.pt"
                
                if not os.path.exists(model_path):
                    return {"success": False, "error": f"Model not found: {model_path}"}
                if not os.path.exists(encoder_path):
                    return {"success": False, "error": f"Encoder not found: {encoder_path}"}
                
                predictors[predictor_key] = HybridSentimentPredictor(
                    encoder_path, 
                    classifier_path=model_path,
                    config_path='configs/config.yaml'
                )
            else:
                model_path = f"results/models/deep_learning/{request.dataset}/{request.encoder}/{request.encoder}_best.pt"
                
                if not os.path.exists(model_path):
                    return {"success": False, "error": f"Model not found: {model_path}"}
                
                predictors[predictor_key] = EndToEndPredictor(
                    model_path=model_path,
                    config_path='configs/config.yaml'
                )
        
        predictor = predictors[predictor_key]
        result = predictor.predict(request.text)

        print(f"PREDICTION RESULT:")
        print(f"   Raw result: {result}")
        print(f"   Sentiment: {result.get('sentiment')}")
        print(f"   Confidence: {result.get('confidence')}")
        print(f"   Probabilities: {result.get('probabilities')}\n")
        
        return {"success": True, "prediction": result}
    
    except Exception as e:
        print(f"[X] PREDICTION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/compare")
async def compare_all_models(request: CompareRequest):
    """Compare text across all available models for a dataset."""
    print(f"\n🔵 COMPARISON REQUEST:")
    print(f"   Text: {request.text}")
    print(f"   Dataset: {request.dataset}\n")
    
    try:
        results = []
        
        # get available models
        models_response = await get_available_models()
        available_models = models_response.get('models', [])
        
        # filter by dataset (or all if "all" is selected)
        if request.dataset.lower() == 'all':
            dataset_models = available_models
            print(f"Comparing across ALL datasets")
        else:
            dataset_models = [m for m in available_models if m['dataset'] == request.dataset]
            print(f"Found {len(dataset_models)} models for {request.dataset}")
        
        for model_info in dataset_models:
            try:
                # create predictor key
                if model_info['type'] == 'hybrid':
                    predictor_key = f"{model_info['dataset']}_{model_info['type']}_{model_info['encoder']}_{model_info['classifier']}"
                    model_name = f"{model_info['encoder'].upper()} + {model_info['classifier'].upper()}"
                    
                    # load predictor if not cached
                    if predictor_key not in predictors:
                        encoder_path = f"results/models/deep_learning/{model_info['dataset']}/{model_info['encoder']}/{model_info['encoder']}_best.pt"
                        predictors[predictor_key] = HybridSentimentPredictor(
                            encoder_path,
                            classifier_path=model_info['path'],
                            config_path='configs/config.yaml'
                        )
                else:
                    predictor_key = f"{model_info['dataset']}_{model_info['type']}_{model_info['encoder']}"
                    model_name = f"{model_info['encoder'].upper()} (End-to-End)"
                    
                    # load predictor if not cached
                    if predictor_key not in predictors:
                        predictors[predictor_key] = EndToEndPredictor(
                            model_path=model_info['path'],
                            config_path='configs/config.yaml'
                        )
                
                # get prediction
                predictor = predictors[predictor_key]
                prediction = predictor.predict(request.text)
                
                results.append({
                    "model": model_name,
                    "dataset": model_info['dataset'],
                    "sentiment": prediction['sentiment'],
                    "confidence": prediction['confidence']
                })
                
                print(f"✓ {model_name} ({model_info['dataset']}): {prediction['sentiment']} ({prediction['confidence']:.2%})")
                
            except Exception as e:
                print(f"✗ Error with {model_info}: {e}")
                continue
        
        return {"success": True, "comparisons": results}
    
    except Exception as e:
        print(f"[X] COMPARISON ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/models/available")
async def get_available_models():
    models = []
    results_dir = "results/models"
    
    if os.path.exists(results_dir):
        dl_path = os.path.join(results_dir, "deep_learning")
        if os.path.exists(dl_path):
            for dataset in os.listdir(dl_path):
                dataset_path = os.path.join(dl_path, dataset)
                if os.path.isdir(dataset_path):
                    for encoder in os.listdir(dataset_path):
                        encoder_path = os.path.join(dataset_path, encoder)
                        if os.path.isdir(encoder_path):
                            e2e_model = os.path.join(encoder_path, f"{encoder}_best.pt")
                            if os.path.exists(e2e_model):
                                models.append({
                                    "dataset": dataset,
                                    "type": "end-to-end",
                                    "encoder": encoder,
                                    "path": e2e_model
                                })
                            
                            hybrid_dir = os.path.join(results_dir, "classical_ml", dataset, encoder)
                            if os.path.exists(hybrid_dir):
                                for clf_file in os.listdir(hybrid_dir):
                                    if clf_file.endswith('.pkl'):
                                        classifier = clf_file.replace('.pkl', '')
                                        models.append({
                                            "dataset": dataset,
                                            "type": "hybrid",
                                            "encoder": encoder,
                                            "classifier": classifier,
                                            "path": os.path.join(hybrid_dir, clf_file)
                                        })
    
    return {"success": True, "models": models}

@app.get("/datasets/available")
async def get_available_datasets():
    """Get list of available datasets."""
    datasets = set()
    results_dir = "results/models"
    
    if os.path.exists(results_dir):
        dl_path = os.path.join(results_dir, "deep_learning")
        if os.path.exists(dl_path):
            for dataset in os.listdir(dl_path):
                dataset_path = os.path.join(dl_path, dataset)
                if os.path.isdir(dataset_path):
                    datasets.add(dataset)
    
    return {"success": True, "datasets": sorted(list(datasets))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)