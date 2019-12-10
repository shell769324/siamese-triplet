import torch
import numpy as np



def fit(train_loader, val_loader, dataloader, refloader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    text_file = open("triplet_50.txt", "w")
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        calcAcc = False
        if epoch%5 == 0:
            calcAcc = True
        val_loss, metrics = test_epoch(val_loader, dataloader, refloader, model, loss_fn, cuda, metrics, calcAcc, text_file)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            #print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, dataloader, refloader, model, loss_fn, cuda, metrics, calcAcc, text_file):
    with torch.no_grad():
        embed_d = 1000
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)
        if calcAcc:
            embeddings = np.zeros((len(dataloader.dataset), embed_d))
            labels = np.zeros(len(dataloader.dataset))
            k = 0
            
            # get the dataset rather than dataloader so the labels are consistent
            class_of_test_data = dataloader.dataset.classes
            idx_to_class = dict()
            for images, target in dataloader:
                if cuda:
                    images = images.cuda()
                embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
                labels[k:k+len(images)] = target.numpy()
                targ = target.numpy()
                for id_ in target:
                    for class_ in class_of_test_data:
                        if id_ == dataloader.dataset.class_to_idx[class_]:
                            idx_to_class[k] = class_
                    k += 1
            ref_embeddings = np.zeros((len(refloader.dataset), embed_d))
            k = 0
            ref_idx_to_class = dict()
            for images, target in refloader:
                if cuda:
                    images = images.cuda()
                ref_embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
                target = target.numpy()
                for id_ in target:
                    for class_ in refloader.dataset.classes:
                        if id_ == refloader.dataset.class_to_idx[class_]:
                            ref_idx_to_class[k] = class_
                    k += 1

            ranks = []
            for i in range(len(embeddings)):
                embedding = embeddings[i]
                rank = 0
                for j, ref_embedding in enumerate(ref_embeddings):
                    if ref_idx_to_class[j] == idx_to_class[i]:
                        t_dist = np.linalg.norm(embedding - ref_embedding)
                        break
                for j, ref_embedding in enumerate(ref_embeddings):
                    # EMBEDDING COMPARISON
                    curr_dist = np.linalg.norm(embedding - ref_embedding)
                    if curr_dist < t_dist:
                        rank += 1
                ranks.append(rank)
            print(ranks)

            print(ranks, val_loss, file = text_file)
    return val_loss, metrics
