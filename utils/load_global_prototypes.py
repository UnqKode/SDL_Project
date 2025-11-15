import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_global_prototypes(path="./data/global_prototypes.pth"):
    chk = torch.load(path, map_location=device,weights_only=False) # default true for security reason
    # false is fine as this pth is trusted
    prototypes = chk["prototypes"].to(device)
    classes = chk["classes"]
    return prototypes, classes