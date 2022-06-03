from os import path

import torch
from torch.optim import Adam
from tqdm import tqdm


def geodesic_distance(a, b, to_sum=True):
    """
    a, b in formal [lat, lon]
    """
    lat1, lon1, lat2, lon2 = torch.stack([a, b], dim=1, out=None).view(-1, 4).t()
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi = torch.deg2rad(lat2 - lat1)
    delta_lambda = torch.deg2rad(lon2 - lon1)
    t = torch.sin(delta_phi / 2) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda / 2) ** 2
    return (6371 * (2 * torch.atan2(torch.sqrt(t), torch.sqrt(1 - t)))).sum() if to_sum else 6371 * (
                2 * torch.atan2(torch.sqrt(t), torch.sqrt(1 - t)))


def train(model, train_data, val_data, learning_rate, epochs, batch_size, model_name):
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    criterion = geodesic_distance
    optimizer = Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch_num in range(epochs):

        total_loss_train = 0
        for train_input, train_label in tqdm(train_dataloader):
            train_label = torch.stack([train_label[0], train_label[1]], dim=1).to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            model.zero_grad()
            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

        total_loss_val = 0
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = torch.stack([val_label[0], val_label[1]], dim=1).to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

        print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
              | Val Loss: {total_loss_val / len(val_data): .3f}')
        if epoch_num % 5 == 4:
            torch.save(model.state_dict(), path.join('models', f'{model_name}_{epoch_num + 1}.pt'))


def evaluate(model, test_data, batch_size):
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    criterion = geodesic_distance

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    distances = list()
    with torch.no_grad():
        for test_input, test_label in test_dataloader:

            test_label = torch.stack([test_label[0], test_label[1]], dim=1).to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            distances.append(criterion(output, test_label, False))
