import gradio as gr
import pandas as pd
import torch
from torchvision.transforms import transforms
from zmq import device

from get_loader import Vocabulary
from model import CNNtoRNN


def predict(img):
    img = transform(img)
    output = model.caption_image(img.unsqueeze(0).to(device), vocab)
    return " ".join(output[1:-1])

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNtoRNN(
        embed_size=256,
        hidden_size=256,
        vocab_size=2994,
        num_layers=1
    )
    print('Loading model weights...')
    model.load_state_dict(torch.load(
        'my_checkpoint.pth.tar', map_location=device)["state_dict"])
    model.to(device)
    model.eval()
    print('Building vocabulary...')
    vocab = Vocabulary(5)
    df = pd.read_csv('flickr8k/captions.txt')
    vocab.build_vocabulary(df['caption'].tolist())

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    print('Creating app...')
    app = gr.Interface(
        fn=predict,
        inputs=gr.Image(shape=(256, 256)),
        outputs="text",
        flagging_options=['yes', 'or', 'no']
    )
    print('done!')

    app.launch(
        share=True,
        debug=True,
    )
