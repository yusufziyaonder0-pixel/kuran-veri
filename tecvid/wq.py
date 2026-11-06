import onnx, numpy as np, sys, os
from onnx import numpy_helper, helper, TensorProto
mode=sys.argv[1]  # int8 | fp16
src='muaalem_fp32_o21.onnx'; base=os.path.dirname(os.path.abspath(src))
m=onnx.load(src, load_external_data=False)
g=m.graph
matmulB=set(n.input[1] for n in g.node if n.op_type=='MatMul')
def arr(t):
    info={e.key:e.value for e in t.external_data}
    dt=onnx.helper.tensor_dtype_to_np_dtype(t.data_type)
    off=int(info.get('offset',0)); ln=int(info['length'])
    return np.memmap(os.path.join(base,info['location']),dtype=dt,mode='r',offset=off,shape=tuple(t.dims) if t.dims else ())
newinit=[]; newnodes=[]; nq=0
for t in list(g.initializer):
    if t.data_location!=TensorProto.EXTERNAL:
        newinit.append(t); continue
    a=np.array(arr(t))
    if a.size<100000 or a.dtype!=np.float32:
        newinit.append(numpy_helper.from_array(a,t.name)); continue
    nq+=1
    if t.name in matmulB and a.ndim==2:
        qmax = 127 if mode=='int8' else 32767; qt = np.int8 if mode=='int8' else np.int16
        sc=np.abs(a).max(axis=0)/qmax; sc[sc==0]=1e-12
        q=np.clip(np.round(a/sc),-qmax,qmax).astype(qt)
        newinit+= [numpy_helper.from_array(q,t.name+'_q'), numpy_helper.from_array(sc.astype(np.float32),t.name+'_s'), numpy_helper.from_array(np.zeros(a.shape[1],qt),t.name+'_z')]
        newnodes.append(helper.make_node('DequantizeLinear',[t.name+'_q',t.name+'_s',t.name+'_z'],[t.name],axis=1))
    else:
        sc=np.float32(max(np.abs(a).max()/32767,1e-12)); q=np.clip(np.round(a/sc),-32767,32767).astype(np.int16)
        newinit+= [numpy_helper.from_array(q,t.name+'_q'), numpy_helper.from_array(np.array(sc,np.float32),t.name+'_s'), numpy_helper.from_array(np.array(0,np.int16),t.name+'_z')]
        newnodes.append(helper.make_node('DequantizeLinear',[t.name+'_q',t.name+'_s',t.name+'_z'],[t.name]))
del g.initializer[:]; g.initializer.extend(newinit)
nodes=list(g.node); del g.node[:]; g.node.extend(newnodes+nodes)
out=f'muaalem_{mode}.onnx'
onnx.save_model(m,out,save_as_external_data=True,all_tensors_to_one_file=True,location=f'muaalem_{mode}.data',size_threshold=1024)
print('converted',nq)
