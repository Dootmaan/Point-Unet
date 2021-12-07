import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.instancenorm import InstanceNorm3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Encoder(nn.Module):
#     """
#     Encoder.
#     shift to only output the feature map
#     """

#     def __init__(self, encoded_image_size=14):
#         super(Encoder, self).__init__()
#         self.enc_image_size = encoded_image_size

#         resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

#         # Remove linear and pool layers (since we're not doing classification)
#         modules = list(resnet.children())[:-2]
#         self.resnet = nn.Sequential(*modules)

#         # Resize image to fixed size to allow input images of variable size
#         self.adaptive_pool = nn.AdaptiveAvgPool3d((encoded_image_size, encoded_image_size))

#         self.fine_tune()

#     def forward(self, images):
#         """
#         Forward propagation.
#         :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
#         :return: encoded images
#         """
#         feature_map = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
#         #out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
#         #out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
#         return feature_map

#     def fine_tune(self, fine_tune=False):
#         """
#         Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
#         :param fine_tune: Allow?
#         """
#         for p in self.resnet.parameters():
#             p.requires_grad = False
#         # If fine-tuning, only fine-tune convolutional blocks 2 through 4
#         for c in list(self.resnet.children())[5:]:
#             for p in c.parameters():
#                 p.requires_grad = fine_tune


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv3d(in_channels=2,
                              out_channels=1,
                              kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        conv = conv.repeat(1, x.size()[1], 1, 1, 1)
        att = torch.sigmoid(conv)
        return x * att

    def agg_channel(self, x, pool="max"):
        b, c, d, h, w = x.size()
        x = x.view(b, c, d * h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, d, h, w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio=1):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in /
                                     float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in))

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3], x.size()[4])
        avg_pool = F.avg_pool3d(x, kernel)
        max_pool = F.max_pool3d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        out = sig_pool.repeat(1, 1, kernel[0], kernel[1], kernel[2])
        return x * out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MultiScaleConv, self).__init__()
        self.conv3x3 = nn.Conv3d(in_ch, out_ch, 3, 1, 1, 1)
        self.conv5x5 = nn.Conv3d(in_ch, out_ch, 3, 1, 2, 2)
        self.conv7x7 = nn.Conv3d(in_ch, out_ch, 3, 1, 3, 3)
        self.norm = InstanceNorm3d(3 * out_ch)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x3 = self.conv7x7(x)
        return self.relu(self.norm(torch.cat((x1, x2, x3), dim=1)))


class SaliencyAttentionNet(nn.Module):
    def __init__(self, in_ch=1):
        super(SaliencyAttentionNet, self).__init__()
        self.pool0 = nn.MaxPool3d(2)

        self.conv1 = DoubleConv(in_ch, 16)
        self.up0 = nn.ConvTranspose3d(16, 16, 2, stride=2)

        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = DoubleConv(32, 32)
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = DoubleConv(32, 32)

        self.multiconv1 = MultiScaleConv(32, 32)
        self.multiconv2 = MultiScaleConv(32, 32)
        self.multiconv3 = MultiScaleConv(32, 32)

        self.sca1 = torch.nn.Sequential(
            ChannelAttention(17),
            SpatialAttention(5),
        )

        self.sca2 = torch.nn.Sequential(
            ChannelAttention(64),
            SpatialAttention(5),
        )
        self.sca_up4 = nn.Sequential(
            nn.ConvTranspose3d(96, 96, 2, stride=2),
            DoubleConv(96, 96),
            nn.ConvTranspose3d(96, 96, 2, stride=2),
        )
        self.squeeze = DoubleConv(96 * 3, 64)
        self.sca_up5 = nn.ConvTranspose3d(96, 96, 2, stride=2)
        self.sca_upall = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 2, stride=2),
            DoubleConv(64, 64),
            nn.ConvTranspose3d(64, 64, 2, stride=2),
        )

        self.conv9 = DoubleConv(64, 24)
        self.conv10 = nn.Sequential(nn.Conv3d(17 + 24, 16, 3, 1, 1),
                                    nn.Conv3d(16, 1, 1, 1, 0))

    def forward(self, x):
        p0 = self.pool0(x)
        c1 = self.conv1(p0)

        sca_cat0 = torch.cat((x, self.up0(c1)), dim=1)
        sca0_out = self.sca1(sca_cat0)

        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        c4m = self.multiconv1(c4)
        c5m = self.multiconv2(c3)
        c6m = self.multiconv3(c2)

        c4u = self.sca_up4(c4m)
        c5u = self.sca_up5(c5m)

        sca_cat1 = torch.cat((c4u, c5u, c6m), dim=1)
        sca_cat1 = self.squeeze(sca_cat1)
        sca1_out = self.sca2(sca_cat1)
        sca1_out = self.sca_upall(sca1_out)

        c9 = self.conv9(sca1_out)
        c10 = self.conv10(torch.cat((c9, sca0_out), dim=1))

        return nn.Sigmoid()(c10)

if __name__=="__main__":
    from thop import clever_format
    from thop import profile
    model=SaliencyAttentionNet()
    inputx=torch.randn(1, 1, 1,128,192, 192)
    
    flops, params = profile(model, inputs=inputx)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops,params)

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())

    print(total_params, 'total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(total_trainable_params,'training parameters.')

# class DecoderWithAttention(nn.Module):
#     """
#     Decoder.
#     shift to sca attention
#     """

#     def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,encoder_out_shape=[1,2048,8,8], K=512,encoder_dim=2048, dropout=0.5):
#         """
#         :param attention_dim: size of attention network
#         :param embed_dim: embedding size
#         :param decoder_dim: size of decoder's RNN
#         :param vocab_size: size of vocabulary
#         :param encoder_dim: feature size of encoded images
#         :param dropout: dropout
#         """
#         super(DecoderWithAttention, self).__init__()

#         self.encoder_dim = encoder_dim
#         self.attention_dim = attention_dim
#         self.embed_dim = embed_dim
#         self.decoder_dim = decoder_dim
#         self.vocab_size = vocab_size
#         self.dropout = dropout

#         self.Spatial_attention = Spatial_attention(encoder_out_shape, decoder_dim, K)  # attention network
#         self.Channel_wise_attention = Channel_wise_attention(encoder_out_shape, decoder_dim, K) # ATTENTION
#         self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
#         self.dropout = nn.Dropout(p=self.dropout)
#         self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
#         self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
#         self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
#         self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
#         self.sigmoid = nn.Sigmoid()
#         self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
#         self.init_weights()  # initialize some layers with the uniform distribution
#         self.AvgPool = nn.AvgPool3d(8)
#     def init_weights(self):
#         """
#         Initializes some parameters with values from the uniform distribution, for easier convergence.
#         """
#         self.embedding.weight.data.uniform_(-0.1, 0.1)
#         self.fc.bias.data.fill_(0)
#         self.fc.weight.data.uniform_(-0.1, 0.1)

#     def load_pretrained_embeddings(self, embeddings):
#         """
#         Loads embedding layer with pre-trained embeddings.
#         :param embeddings: pre-trained embeddings
#         """
#         self.embedding.weight = nn.Parameter(embeddings)

#     def fine_tune_embeddings(self, fine_tune=True):
#         """
#         Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
#         :param fine_tune: Allow?
#         """
#         for p in self.embedding.parameters():
#             p.requires_grad = fine_tune

#     def init_hidden_state(self, encoder_out):
#         """
#         Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
#         :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
#         :return: hidden state, cell state
#         """
#         mean_encoder_out = self.AvgPool(encoder_out).squeeze(-1).squeeze(-1)
#         h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
#         c = self.init_c(mean_encoder_out)
#         return h, c

#     def forward(self, encoder_out, encoded_captions, caption_lengths):
#         """
#         Forward propagation.
#         :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
#         :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
#         :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
#         :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
#         """

#         batch_size = encoder_out.size(0)
#         encoder_dim = encoder_out.size(-1)
#         vocab_size = self.vocab_size

#         # Flatten image
#         # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
#         # num_pixels = encoder_out.size(1)

#         # Sort input data by decreasing lengths; why? apparent below
#         caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
#         encoder_out = encoder_out[sort_ind]
#         encoded_captions = encoded_captions[sort_ind]

#         # Embedding
#         embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

#         # Initialize LSTM state
#         h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

#         # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
#         # So, decoding lengths are actual lengths - 1
#         decode_lengths = (caption_lengths - 1).tolist()

#         # Create tensors to hold word predicion scores and alphas
#         predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)#需要更改形状？
#         #alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)#需要更改形状

#         # At each time-step, decode by
#         # attention-weighing the encoder's output based on the decoder's previous hidden state output
#         # then generate a new word in the decoder with the previous word and the attention weighted encoding
#         for t in range(max(decode_lengths)):
#             batch_size_t = sum([l > t for l in decode_lengths])
#             # attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
#             #                                                     h[:batch_size_t])
#             #channel-spatial模式attention
#             #channel_wise
#             attention_weighted_encoding, beta = self.Channel_wise_attention(encoder_out[:batch_size_t],h[:batch_size_t])
#             #spatial
#             attention_weighted_encoding, alpha = self.Spatial_attention(attention_weighted_encoding[:batch_size_t],h[:batch_size_t])
#             #对attention_weighted_encoding降维
#             attention_weighted_encoding = attention_weighted_encoding.view(attention_weighted_encoding.shape[0],2048,8,8)
#             attention_weighted_encoding = self.AvgPool(attention_weighted_encoding)
#             attention_weighted_encoding = attention_weighted_encoding.squeeze(-1).squeeze(-1)
#             # gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
#             # attention_weighted_encoding = gate * attention_weighted_encoding
#             h, c = self.decode_step(
#                 torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
#                 (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
#             preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
#             predictions[:batch_size_t, t, :] = preds
#             #alphas[:batch_size_t, t, :] = alpha

#         return predictions, encoded_captions, decode_lengths, sort_ind
