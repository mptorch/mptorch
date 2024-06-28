import os
import torch
import torch.nn as nn
from torch import optim
from model import VisionTransformer
from data_loader import get_loader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from mptorch import FloatingPoint
import mptorch.quant as qpt
from tqdm import tqdm


class Solver(object):
    def __init__(self, args):
        self.args = args

        # Get data loaders
        self.train_loader, self.test_loader = get_loader(args)

        rounding = "nearest"
        fp_format = FloatingPoint(
            exp=args.expMac, man=args.manMac, subnormals=True, saturate=False
        )
        quant_fp = lambda x: qpt.float_quantize(
            x,
            exp=args.expWeight,
            man=args.manWeight,
            rounding=rounding,
            saturate=False,
        )
        layer_formats = qpt.QAffineFormats(
            fwd_mac=(fp_format),
            fwd_rnd=rounding,
            bwd_mac=(fp_format),
            bwd_rnd=rounding,
            weight_quant=quant_fp,
            input_quant=quant_fp,
            grad_quant=quant_fp,
            bias_quant=quant_fp,
        )
        # layer_formats = None

        # Create object of the Vision Transformer
        self.model = VisionTransformer(
            n_channels=self.args.n_channels,
            embed_dim=self.args.embed_dim,
            n_layers=self.args.n_layers,
            n_attention_heads=self.args.n_attention_heads,
            forward_mul=self.args.forward_mul,
            image_size=self.args.image_size,
            patch_size=self.args.patch_size,
            n_classes=self.args.n_classes,
            dropout=self.args.dropout,
            formats=layer_formats,
        )

        # Push to GPU
        if self.args.is_cuda:
            print("Using GPU")
            self.model = self.model.cuda()
        else:
            print("Cuda not available.")

        # Display Vision Transformer
        print("--------Network--------")
        print(self.model)

        # Option to load pretrained model
        if args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.model_path, "ViT_model.pt"))
            )

        # Training loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Arrays to record training progression
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def test_dataset(self, loader):
        # Set Vision Transformer to evaluation mode
        self.model.eval()

        # Arrays to record all labels and logits
        all_labels = []
        all_logits = []

        # Testing loop
        for x, y in loader:
            if self.args.is_cuda:
                x = x.cuda()

            # Avoid capturing gradients in evaluation time for faster speed
            with torch.no_grad():
                logits = self.model(x)

            all_labels.append(y)
            all_logits.append(logits.cpu())

        # Convert all captured variables to torch
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)
        all_pred = all_logits.max(1)[1]

        # Compute loss, accuracy and confusion matrix
        loss = self.loss_fn(all_logits, all_labels).item()
        acc = accuracy_score(y_true=all_labels, y_pred=all_pred)
        cm = confusion_matrix(
            y_true=all_labels, y_pred=all_pred, labels=range(self.args.n_classes)
        )

        return acc, cm, loss

    def test(self, train=True):
        if train:
            # Test using train loader
            acc, _, loss = self.test_dataset(self.train_loader)
            print(f"Train acc: {acc:.2%}\tTrain loss: {loss:.4f}")

        # Test using test loader
        acc, _, loss = self.test_dataset(self.test_loader)
        print(f"Test acc: {acc:.2%}\tTest loss: {loss:.4f}")

        return acc, loss

    def train(self):
        iters_per_epoch = len(self.train_loader)

        # Define optimizer for training the model
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=1e-3
        )

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group["lr"]

        # scheduler for linear warmup of lr and then cosine decay
        linear_warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / self.args.warmup_epochs,
            end_factor=1.0,
            total_iters=self.args.warmup_epochs - 1,
            last_epoch=-1,
            verbose=True,
        )
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.args.epochs - self.args.warmup_epochs,
            eta_min=1e-5,
            verbose=True,
        )

        # Variable to capture best test accuracy
        best_acc = 0
        init_scale = 256.0

        # Set up loss scaling
        if self.args.is_cuda:
            scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
        else:
            scaler = None

        # Training loop
        for epoch in range(self.args.epochs):
            tq = tqdm(total=len(self.train_loader) * self.args.batch_size)

            tq.set_description(f"Epoch {epoch}, lr {get_lr(optimizer):.3f}")
            # Set model to training mode
            self.model.train()

            # Arrays to record epoch loss and accuracy
            train_epoch_loss = []
            train_epoch_accuracy = []

            # Loop on loader
            for i, (x, y) in enumerate(self.train_loader):

                # Push to GPU
                if self.args.is_cuda:
                    x, y = x.cuda(), y.cuda()

                # Get output logits from the model
                logits = self.model(x)

                # Compute training loss
                loss = self.loss_fn(logits, y)

                # Updating the model
                optimizer.zero_grad()

                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                tq.update(self.args.batch_size)
                # Batch metrics
                batch_pred = logits.max(1)[1]
                batch_accuracy = (y == batch_pred).float().mean()
                train_epoch_loss += [loss.item()]
                train_epoch_accuracy += [batch_accuracy.item()]

                # Log training progress
                if scaler is None:
                    tq.set_postfix(
                        batch_acc="{:.5f}".format(batch_accuracy),
                        batch_loss="{:.5f}".format(loss),
                    )
                else:
                    tq.set_postfix(
                        batch_acc="{:.5f}".format(batch_accuracy),
                        batch_loss="{:.5f}".format(loss),
                        loss_scale="{:.5f}".format(scaler.get_scale()),
                    )

            tq.close()
            # Test the test set after every epoch
            test_acc, test_loss = self.test(
                train=((epoch + 1) % 25 == 0)
            )  # Test training set every 25 epochs

            # Capture best test accuracy
            best_acc = max(test_acc, best_acc)
            print(f"Best test acc: {best_acc:.2%}\n")

            # Save model
            torch.save(
                self.model.state_dict(),
                os.path.join(self.args.model_path, "ViT_model.pt"),
            )

            # Update learning rate using schedulers
            if epoch < self.args.warmup_epochs:
                linear_warmup.step()
            else:
                cos_decay.step()

            # Update training progression metric arrays
            self.train_losses += [sum(train_epoch_loss) / iters_per_epoch]
            self.test_losses += [test_loss]
            self.train_accuracies += [sum(train_epoch_accuracy) / iters_per_epoch]
            self.test_accuracies += [test_acc]

    def plot_graphs(self):
        # Plot graph of loss values
        plt.plot(self.train_losses, color="b", label="Train")
        plt.plot(self.test_losses, color="r", label="Test")

        plt.ylabel("Loss", fontsize=18)
        plt.yticks(fontsize=16)
        plt.xlabel("Epoch", fontsize=18)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=15, frameon=False)

        # plt.show()  # Option to view graph while training
        plt.savefig(
            os.path.join(self.args.output_path, "graph_loss.png"), bbox_inches="tight"
        )
        plt.close("all")

        # Plot graph of accuracies
        plt.plot(self.train_accuracies, color="b", label="Train")
        plt.plot(self.test_accuracies, color="r", label="Test")

        plt.ylabel("Accuracy", fontsize=18)
        plt.yticks(fontsize=16)
        plt.xlabel("Epoch", fontsize=18)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=15, frameon=False)

        # plt.show()  # Option to view graph while training
        plt.savefig(
            os.path.join(self.args.output_path, "graph_accuracy.png"),
            bbox_inches="tight",
        )
        plt.close("all")
