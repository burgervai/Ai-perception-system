import onnx
from pathlib import Path
p = Path('ml/checkpoints/model.fp16.onnx')
model = onnx.load(p)
print('exists', p.exists())
print('graph outputs:', [(o.name, o.type.tensor_type.elem_type) for o in model.graph.output])
vi = {vi.name: vi for vi in model.graph.value_info}
for name in ['mean']:
    if name in vi:
        print(name, 'value_info type', vi[name].type.tensor_type.elem_type)
    else:
        print(name, 'not in value_info')
for n in model.graph.node:
    if 'mean' in n.output or 'mean' in n.input:
        print('node', n.name, n.op_type, 'inputs', list(n.input), 'outputs', list(n.output))
        for a in n.attribute:
            print('  attr', a.name, a)
