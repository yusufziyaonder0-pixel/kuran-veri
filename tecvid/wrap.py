import onnx, sys
from onnx import helper as h, TensorProto as TP
mode=sys.argv[1]
m=onnx.load(f'muaalem_{mode}.onnx', load_external_data=False)
g=m.graph
# rename model inputs
fe,ma=g.input[0].name,g.input[1].name
nodes=[
 h.make_node('ReduceMean',['features','ax1'],['mu'],keepdims=1),
 h.make_node('Sub',['features','mu'],['xc']),
 h.make_node('Mul',['xc','xc'],['xc2']),
 h.make_node('ReduceSum',['xc2','ax1'],['ss'],keepdims=1),
 h.make_node('Shape',['features'],['shp']),
 h.make_node('Gather',['shp','i1'],['T'],axis=0),
 h.make_node('Cast',['T'],['Tf'],to=TP.FLOAT),
 h.make_node('Sub',['Tf','one_f'],['Tm1']),
 h.make_node('Div',['ss','Tm1'],['var']),
 h.make_node('Add',['var','eps'],['var2']),
 h.make_node('Sqrt',['var2'],['sd']),
 h.make_node('Div',['xc','sd'],['xn']),
 h.make_node('Div',['T','two_i'],['Th']),
 h.make_node('Mul',['Th','two_i'],['T2']),
 h.make_node('Unsqueeze',['T2','ax0'],['T2u']),
 h.make_node('Slice',['xn','zero_i1','T2u','ax1'],['xs']),
 h.make_node('Reshape',['xs','rs'],[fe]),
 h.make_node('Shape',[fe],['shp2']),
 h.make_node('Slice',['shp2','zero_i1','two_i1'],['shp2b']),
 h.make_node('ConstantOfShape',['shp2b'],[ma],value=h.make_tensor('v',TP.INT64,[1],[1])),
]
inits=[h.make_tensor('ax1',TP.INT64,[1],[1]),h.make_tensor('ax0',TP.INT64,[1],[0]),h.make_tensor('i1',TP.INT64,[],[1]),
 h.make_tensor('one_f',TP.FLOAT,[],[1.0]),h.make_tensor('eps',TP.FLOAT,[],[1e-7]),h.make_tensor('two_i',TP.INT64,[],[2]),
 h.make_tensor('zero_i1',TP.INT64,[1],[0]),h.make_tensor('two_i1',TP.INT64,[1],[2]),h.make_tensor('rs',TP.INT64,[3],[0,-1,160])]
out=g.output[0].name
post=[h.make_node('LogSoftmax',[out],['log_probs'],axis=-1),
      h.make_node('Shape',['log_probs'],['shp3']), h.make_node('Slice',['shp3','i1_1','two_i1'],['ulen'])]
inits.append(h.make_tensor('i1_1',TP.INT64,[1],[1]))
allnodes=nodes+list(g.node)+post
del g.node[:]; g.node.extend(allnodes); g.initializer.extend(inits)
del g.input[:]; g.input.extend([h.make_tensor_value_info('features',TP.FLOAT,['N','T',80]),h.make_tensor_value_info('features_length',TP.INT64,['N'])])
del g.output[:]; g.output.extend([h.make_tensor_value_info('log_probs',TP.FLOAT,['N','U',43]),h.make_tensor_value_info('ulen',TP.INT64,[1])])
# features_length unused -> add identity to keep it
for k,v in {'model_type':'fire-red-asr-2-ctc','subsampling_factor':'4','cmvn_mean':','.join(['0']*80),'cmvn_inv_stddev':','.join(['1']*80),'comment':'Quran Muaalem (obadx/muaalem-model-v3_2, MIT) wrapped for sherpa-onnx; per-utterance CMVN + stride-2 stacking inside'}.items():
    p=m.metadata_props.add(); p.key=k; p.value=v
onnx.save_model(m, f'mua_{mode}.onnx')
print('ok')
