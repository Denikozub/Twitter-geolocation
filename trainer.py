import torch
from torch.optim import Adam
from tqdm import tqdm


def geodesic_distance(a, b):
    """
    a, b in formal [lat, lon]
    """
    lat1, lon1, lat2, lon2 = torch.stack([a, b], dim=1, out=None).view(-1, 4).t()
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi = torch.deg2rad(lat2 - lat1)
    delta_lambda = torch.deg2rad(lon2 - lon1)
    t = torch.sin(delta_phi / 2) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda / 2) ** 2
    return (6371 * (2 * torch.atan2(torch.sqrt(t), torch.sqrt(1 - t)))).sum()


def train(model, train_data, val_data, learning_rate, criterion, epochs, batch_size, AE=False):
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch_num in range(epochs):

        total_loss_train = 0
        for train_input, train_label in tqdm(train_dataloader):
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            model.zero_grad()
            if AE:
                output, train_label = model(input_id, mask)
            else:
                output = model(input_id, mask)
                train_label = torch.stack([train_label[0], train_label[1]], dim=1).to(device)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

        total_loss_val = 0
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                if AE:
                    output, val_label = model(input_id, mask)
                else:
                    output = model(input_id, mask)
                    val_label = torch.stack([val_label[0], val_label[1]], dim=1).to(device)
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

        print(f'Epochs: {epoch_num + 1}')
        print(f'Train Loss: {total_loss_train / len(train_data)}')
        print(f'Val Loss: {total_loss_val / len(val_data)}')
        if epoch_num % 5 == 4:
            model.save(epoch_num + 1)


def evaluate(model, test_data, batch_size, criterion, AE=False):
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_loss_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:

            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            if AE:
                output, test_label = model(input_id, mask)
            else:
                output = model(input_id, mask)
                test_label = torch.stack([test_label[0], test_label[1]], dim=1).to(device)
            total_loss_test += criterion(output, test_label)

    print(f'Test Loss: {total_loss_test / len(test_data)}')
