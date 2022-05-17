import torch
import torch.nn as nn
import torchvision.models as models

#----------------------------------------------------------------------------

class EncoderCNN(nn.Module):
    def __init__(self, embed_size) -> None:
        super().__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        for param in self.inception.parameters():
            param.requires_grad = False
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU(True)
        self.times = []
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, imgs):
        features = self.inception(imgs)
        return self.dropout(self.relu(features))

#----------------------------------------------------------------------------

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.LSTM = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        embbedings = self.dropout(self.embed(captions))
        # unsqueeze(0) 添加时间维度seq_len
        embbedings = torch.cat([features.unsqueeze(0), embbedings], dim=0)
        hiddens, _ = self.LSTM(embbedings)
        outputs = self.linear(hiddens)
        return outputs

#----------------------------------------------------------------------------

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super().__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, imgs, captions):
        features = self.encoderCNN(imgs)
        outputs = self.decoderRNN(features, captions)
        return outputs
    
    def caption_image(self, img, vocab, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(img).unsqueeze(0)
            states = None

            for _ in range(max_length):
                # 逐个预测
                h, states = self.decoderRNN.LSTM(x, states)
                output = self.decoderRNN.linear(h.squeeze(0))
                predicted = output.argmax(1)
                # 预测的值作为下一次预测的输入
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocab.itos[predicted.item()] == '<EOS>':
                    break

        return [vocab.itos[idx] for idx in result_caption]
