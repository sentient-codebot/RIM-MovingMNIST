from argument_parser import argument_parser
from datasets import setup_dataloader


args = argument_parser()

foo, val_loader, bar = setup_dataloader(args)

print(len(val_loader.dataset))
t = next(iter(val_loader))
print(t[3].shape)
...