import torch
from sample_torch_model import FeedForwardNet, download_mnist_datasets

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Eg prediction: Tensor (1, 10) -> [[0.1, 0.1, 0.01, 0.02, 0.5, 0.03..., 0.07]]
        predicted_idx = predictions[0].argmax(0)
        predicted_class = class_mapping[predicted_idx]
        expected_class = class_mapping[target]
    return predicted_class, expected_class


if __name__ == "__main__":
    # load the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet_lr001.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load the MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    # get a sample from the validation dataset for inference
    input, target = validation_data[0][0], validation_data[0][1]

    # make an inference
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)

    print(f"Predicted class: {predicted}\nExpected class: {expected}")