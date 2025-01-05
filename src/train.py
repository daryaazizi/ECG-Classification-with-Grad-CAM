from torch import nn
from tqdm import tqdm


def train_one_epoch(
    model, dataloader, optimizer, loss_fn, epoch_number, device="cuda"
) -> None:
    model.train()
    model.to(device)

    dataloader_iterator = tqdm(dataloader, colour="green", leave=True)
    for data, targets in dataloader_iterator:
        data = data.to(device)
        targets = targets.to(device)
        loss = perform_training_step(model, optimizer, loss_fn, data, targets)

        dataloader_iterator.set_description(f"[EPOCH {epoch_number}]")
        dataloader_iterator.set_postfix(batch_loss=loss.item())


def perform_training_step(model, optimizer, loss_fn, data, targets):
    predictions = model(data)
    if isinstance(loss_fn, nn.BCELoss):
        loss = loss_fn(predictions, targets.unsqueeze(1).float())
    else:
        loss = loss_fn(predictions, targets.long())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss
