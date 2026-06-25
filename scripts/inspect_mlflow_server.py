from mlflow.tracking import MlflowClient
import json
import os

TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI','http://localhost:5000')
client = MlflowClient(tracking_uri=TRACKING_URI)
models = client.search_model_versions("name='pytorch_model'")
out = []
for mv in models:
    out.append({
        'name': mv.name,
        'version': mv.version,
        'run_id': mv.run_id,
        'current_stage': getattr(mv, 'current_stage', None),
        'status': getattr(mv, 'status', None),
        'status_message': getattr(mv, 'status_message', None)
    })
print(json.dumps({'tracking_uri': TRACKING_URI, 'results': out}, indent=2))
