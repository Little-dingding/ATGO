import torch
import logging as log
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def save_model(net, optim, loss, models_path):
    torch.save({
        'state_dict': net.state_dict(),
        'optimizer': optim.state_dict(),
        'loss': loss},
        models_path)
    log.info('save model {} success'.format(models_path))


def resume_model(net, resume_model_name=None):
    log.info('resuming model...')
    models = {}
    if len(resume_model_name) > 0:
        model_name = '{}'.format(resume_model_name)
    else:
        log.info('model param is None...')
        index = sorted(models)[-1]
        model_name = models[index]
    model_dict = torch.load(model_name)
    net.load_state_dict(model_dict['state_dict'])
    optim_state = model_dict['optimizer']
    loss = model_dict['loss']
    log.info('finish to resume model {}.'.format(model_name))
    return optim_state, loss