import onnxruntime as ort
import numpy as np
from pathlib import Path
p = Path('ml/checkpoints/model.fp16.onnx')
print('model exists', p.exists())
sess = ort.InferenceSession(str(p), providers=['CPUExecutionProvider'])
print('inputs', [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print('outputs', [(o.name, o.shape, o.type) for o in sess.get_outputs()])
x = np.zeros((1,3,224,224), dtype=np.float32)
outs = sess.run([o.name for o in sess.get_outputs()], {sess.get_inputs()[0].name: x})
for idx, out in enumerate(outs):
    print(idx, out.shape, out.dtype)
    print(out.reshape(-1)[:10])
