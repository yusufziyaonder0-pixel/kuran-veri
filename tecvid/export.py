import torch, numpy as np, os
os.makedirs('onnx',exist_ok=True)
from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
m=Wav2Vec2BertForMultilevelCTC.from_pretrained('obadx/muaalem-model-v3_2').eval()
class W(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s, feats, mask):
        h=s.m.wav2vec2_bert(feats, attention_mask=mask, return_dict=False)[0]
        return s.m.level_to_lm_head['phonemes'](h)
w=W(m).eval()
x=torch.randn(1,200,160); a=torch.ones(1,200,dtype=torch.long)
with torch.no_grad():
    torch.onnx.export(w,(x,a),'onnx/muaalem.onnx',input_names=['feats','mask'],output_names=['logits'],
        dynamic_axes={'feats':{1:'T'},'mask':{1:'T'},'logits':{1:'U'}},opset_version=17,dynamo=False)
import onnx, shutil
mm=onnx.load('onnx/muaalem.onnx')
onnx.save_model(mm,'muaalem_fp32.onnx',save_as_external_data=True,all_tensors_to_one_file=True,location='muaalem_fp32.data')
shutil.rmtree('onnx')
from onnx import version_converter
m2=version_converter.convert_version(onnx.load('muaalem_fp32.onnx',load_external_data=False),21)
onnx.save_model(m2,'muaalem_fp32_o21.onnx')
print('done')
