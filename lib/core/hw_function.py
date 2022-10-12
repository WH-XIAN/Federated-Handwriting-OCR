from  __future__ import  absolute_import
import time
import lib.config.utils.utils as utils
import torch
import chardet
import editdistance as ed

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, logger, writer_dict=None, output_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (inp, labels, idx) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        # labels = utils.get_batch_label(dataset, idx)
        inp = inp.to(device)
        
        # print('cat inp shape ', inp.shape)
       
        # print('inp shape {0} and its idx {1}'.format(inp[0].shape, idx))
        # inference
        preds = model(inp).cpu()

        # compute loss
        batch_size = inp.size(0)
        text, length = converter.encode(labels)                    # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        # print('loss length', length)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
        loss = criterion(preds, text, preds_size, length)
        
        # loss = topk(loss, k)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            #print(msg)
            logger.info(msg)


            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, logger, writer_dict, output_dict):

    losses = AverageMeter()
    model.eval()

    n_correct = 0
    ed_dist = 0
    len_test_sample = 0
    with torch.no_grad():
        for i, (inp, labels, idx) in enumerate(val_loader):

            
            # labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)
            # valid_num = valid_num.to(device) 

            # inference
            preds = model(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), inp.size(0))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                # print('pred : ==> ', pred , ' === ', 'target ===> ', target)
                # print('pred type ===>', chardet.detect(str.encode(pred)))
                # print('gt type ===>', chardet.detect(str.encode(target)))
                if pred == target.strip():
                    # print('string ===== is working ====>')
                    n_correct += 1
                elif pred.__eq__(target.strip()):
                    # print('string equals should work ====>')
                    n_correct += 1
                '''
                else:
                    print('maybe sth else ====>')
                    print('pred : ==> ', len(pred) , ' === ', 'target ===> ', len(target))
                '''
                ed_dist += ed.eval(pred, target.strip())
                len_test_sample += len(target.strip())
            if (i + 1) % config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

            if i == config.TEST.NUM_TEST_BATCH:
                break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        
        print('pred type ===>', chardet.detect(str.encode(pred)))
        print('gt type ===>', chardet.detect(str.encode(gt)))

        logger.info('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(dataset):
        num_test_sample = len(dataset)

    print("[#correct:{} / #total:{}]".format(n_correct, num_test_sample))
    accuracy = n_correct / float(num_test_sample)
    avg_ed_dist = ed_dist / float(len_test_sample)
    logger.info('Test loss: {:.4f}, accuracy: {:.4f}, editdistance: {:.4f}'.format(losses.avg, accuracy, avg_ed_dist))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy