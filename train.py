from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(EPOCH):
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(input, mask)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(pred, dim=1)
            report = evaluate(y_pred.cpu().data.numpy(), target.cpu().data.numpy(), id2label, output_dict=True)

        print(
            '>> epoch:', e,
            'batch:', b,
            'loss:', round(loss.item(), 5),
            'train_f1:', report['macro avg']['f1-score'],
        )

        torch.save(model.state_dict(), MODEL_DIR + f'{e}.pth')