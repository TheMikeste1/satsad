import torch

def main():
    x = torch.rand(5, 3)
    print(x)
    b = torch.cuda.is_available()
    print(b)


if __name__ == "__main__":
    main()
