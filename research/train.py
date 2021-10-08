
import json
from datetime import datetime

import random
import numpy as np

import sys

import torch
import tqdm


def save_model(model, epoch, step, model_path):
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'step': step,
    }, str(model_path))


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(model, criterion, train_loader, valid_loader, validation, optimizer, device,
          scheduler=None, n_epochs=10, model_path=None, log_path=None):
    if model_path:
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    report_each = 10
    if log_path:
        log = log_path.open('at', encoding='utf8')
    else:
        log = sys.stdout

    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader)))
        tq.set_description('Epoch {}'.format(epoch))
        losses = []
        try:
            mean_loss = 0
            for i_step, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.view(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3],
                                     inputs.shape[4])
                inputs = inputs.to(device)
                targets = targets.view(targets.shape[0] * targets.shape[1], targets.shape[2], targets.shape[3],
                                       targets.shape[4])

                with torch.no_grad():
                    targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(1)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i_step and i_step % report_each == 0:
                    write_event(log, step, loss=mean_loss)

            if scheduler:
                scheduler.step()

            write_event(log, step, loss=mean_loss)
            tq.close()
            save_model(model, epoch + 1, step, model_path)
            valid_metrics = validation(model, criterion, valid_loader, device)
            write_event(log, step, **valid_metrics)

            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save_model(model, epoch, step, model_path)
            print('done.')
            return
