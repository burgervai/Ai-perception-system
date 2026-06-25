from mlflow.tracking import MlflowClient
import json

client = MlflowClient()
models = client.search_model_versions("name='pytorch_model'")
out = []
for mv in models:
    out.append({
        'name': mv.name,
        'version': mv.version,
        'run_id': mv.run_id,
        'creation_timestamp': getattr(mv, 'creation_timestamp', None),
        'current_stage': getattr(mv, 'current_stage', None),
        'status': getattr(mv, 'status', None),
        'status_message': getattr(mv, 'status_message', None),
        'description': getattr(mv, 'description', None),
    })
print(json.dumps(out, indent=2))
