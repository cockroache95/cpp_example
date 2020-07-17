import mxnet as mx
import struct

epoch = 0

IS_USE_GPU = True
gpu_id = 0

prefix = "arcface/model-r100-ii/model"

sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

if IS_USE_GPU:
    ctx = mx.gpu(gpu_id)
else:
    ctx = mx.cpu()

model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
model.bind(data_shapes=[('data', (1, 3, 112, 112))], for_training=False)
model.set_params(arg_params, aux_params)

f = open('arcface.wts', 'w')
f.write('{}\n'.format(len(model.get_params()[0].keys()) + len(model.get_params()[1].keys())))
for k, v in model.get_params()[0].items():
    vr = v.reshape(-1).asnumpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
for k, v in model.get_params()[1].items():
    vr = v.reshape(-1).asnumpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')