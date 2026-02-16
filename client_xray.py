import argparse
import os
import torch
import torchvision
from model import ConvNet2, LoRAConvNet2
from torch import nn
from torch.optim import SGD
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomHorizontalFlip
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast, GradScaler

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

def evaluate(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Accuracy of the network: {accuracy:.2f} %")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_path", type=str, default="chest_xray")
    parser.add_argument("--model_type", type=str, default="full", choices=["full", "lora"])
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--use_amp", type=str, default="True")
    args = parser.parse_args()
    
    # Convert string to boolean
    args.use_amp = args.use_amp.lower() in ['true', '1', 'yes']
    
    epochs = args.epochs
    lr = 0.001 # Reduced LR for deeper model
    
    # (3) initializes NVFlare client API
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    normalized_client_name = client_name.replace("-", "").replace("_", "")
    print(f"Original site name: {client_name}, Normalized: {normalized_client_name}")
    print(f"Model type: {args.model_type}, Use AMP: {args.use_amp}")
    
    # Build data paths based on normalized site name
    site_data_path = os.path.join(args.data_path, normalized_client_name)
    train_dir = os.path.join(site_data_path, "train")
    test_dir = os.path.join(site_data_path, "test")

    if args.model_type == "lora":
        model = LoRAConvNet2(rank=args.lora_rank)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    else:
        model = ConvNet2()
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    
    # Data transforms
    transform = Compose(
        [
            Resize((224, 224)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
        ]
    )

    # Load datasets
    print(f"Loading data for {client_name} from {train_dir}")
    train_set = ImageFolder(root=train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_set = ImageFolder(root=test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # (optional) metrics tracking
    summary_writer = SummaryWriter()

    while flare.is_running():
        # (4) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")
        
        # (5) loads model from NVFlare
        model.load_state_dict(input_model.params)
        model.to(device)
        
        # Reset optimizer every round for stability in FL and use Adam for both
        if args.model_type == "lora":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            
        # Mixed precision scaler
        scaler = GradScaler(enabled=args.use_amp and device.type == "cuda")
        
        # (6) evaluate on received model
        accuracy = evaluate(model, test_loader, device)

        # (optional) Task branch for cross-site evaluation
        if flare.is_evaluate():
            print(f"site = {client_name}, running cross-site evaluation")
            output_model = flare.FLModel(metrics={"accuracy": accuracy})
            flare.send(output_model)
            continue

        model.train()
        steps = epochs * len(train_loader)
        for epoch in range(epochs):
            running_loss = 0.0
            total_preds = []
            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()

                with autocast(enabled=args.use_amp and device.type == "cuda"):
                    predictions = model(images)
                    cost = loss(predictions, labels)

                if scaler.is_enabled():
                    scaler.scale(cost).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    cost.backward()
                    optimizer.step()

                running_loss += cost.item()
                
                # Track prediction distribution
                _, predicted = torch.max(predictions.data, 1)
                total_preds.extend(predicted.cpu().numpy().tolist())

                if i % 10 == 9 or i == len(train_loader) - 1:
                    avg_loss = running_loss / (i % 10 + 1)
                    
                    # Log prediction distribution to monitor collapse
                    p_ratio = sum(total_preds) / len(total_preds) if len(total_preds) > 0 else 0
                    print(f"[{epoch+1}, {i+1:5d}] loss: {avg_loss:.3f} | Pred Ratio (Pneumonia): {p_ratio:.2f}")

                    # Log metrics
                    global_step = input_model.current_round * steps + epoch * len(train_loader) + i
                    summary_writer.add_scalar(tag="loss", scalar=avg_loss, global_step=global_step)
                    running_loss = 0.0
                    total_preds = []

        print(f"Finished Training for {client_name}")

        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
