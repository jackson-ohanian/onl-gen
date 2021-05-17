



# Begin VAE
########################################################
class GAN:
    def __init__(self, dist_name, out_samples, path, latent, conv_ct, n_epochs, im_size, hide_size, fn):
        self.out_samples = out_samples
        self.dist = dist_name
        self.path = path
        self.latent = latent
        self.conv_ct = conv_ct
        self.n_epochs = n_epochs
        self.im_size = im_size
        self.hide_size = hide_size
        self.file_path = os.path.join(self.path,  os.path.join("data",  fn))
        self.batch_size = 1
        self.gen = self.begin_gen()
        self.disc = self.begin_disc()
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train()

    def begin_gen(self):

        layers_gen = torch.nn.Sequential(
            torch.nn.Linear(self.latent, self.im_size**2),
            torch.nn.LeakyReLU(0.2),
            ReshapeLayer(-1, 16, 7, 7),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels = 16, out_channels=16, kernel_size = 3, stride=1, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels = 16, out_channels=8, kernel_size = 3, stride=1, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(in_channels = 8, out_channels=1, kernel_size = 3, stride=1, padding=1),
            torch.nn.Sigmoid()
            #torch.nn.Linear(self.hide_size, self.im_size)
        )

        return Generator(self.latent, self.im_size, layers_gen, self.dist, self.hide_size)

    def begin_disc(self):
        layers_disc = torch.nn.Sequential (
            torch.nn.Conv2d(in_channels = 1, out_channels=2, kernel_size = 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.Conv2d(in_channels = 2, out_channels=4, kernel_size = 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.Conv2d(in_channels = 4, out_channels=8, kernel_size = 3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 200, out_features = 1)
        )
        return Discriminator(self.latent, self.im_size, layers_disc, self.dist, self.hide_size)

    def loss_dis(self, batch_size, batch_data, z):

        loss = torch.nn.BCEWithLogitsLoss()
        ##########keep keep
        fake_imgs = torch.squeeze(self.gen(z), dim=0)
        print(batch_data.shape)
        print(fake_imgs.shape)

        all_imgs = torch.cat((batch_data, fake_imgs), 0)

        all_imgs = all_imgs.view(batch_size*2, 1, self.im_size, self.im_size)
        try1 = self.disc(all_imgs)

        target = torch.cat((torch.ones(batch_size, 1), torch.zeros(int(batch_size), 1)), 0)
        loss_d = 0.5*loss(try1, target)

        return loss_d

    def loss_gen(self, batch_size, z):
        loss = torch.nn.BCEWithLogitsLoss()

        loss_on_gen = self.disc(self.gen(z).view(-1, 1, self.im_size, self.im_size))
        target = torch.ones(int(batch_size/2), 1)

        loss_g = loss(loss_on_gen, target)
        return loss_g

    def train(self, iter_d=1, iter_g=1, batch_size=1, lr=0.0002):

        train_data = parse_images(self.file_path)

        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,

        )

        dopt = optim.Adam(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(self.n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self.latent, device=self._dev) - 1
                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self.loss_dis(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self.loss_gen(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()
            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")

        self.batch_size = 1
        for i in range(self.out_samples):
            z = 2 * torch.rand(data.size()[0], self.latent, device=self._dev) - 1

            with torch.no_grad():
                tmpimg = self.gen(z).detach().cpu().squeeze()
            print(tmpimg.shape)
            print(tmpimg.shape)
            img = torchvision.transforms.functional.to_pil_image(tmpimg[0])
            img.save("out.png")


# Begin Encoder and Decoder
########################################################
class Generator(torch.nn.Module):
    def __init__(self, latent_dim, in_dim, layers, dist, hidden):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.in_dim = in_dim
        self.layers = layers
        self.dist = dist
        self.hidden = hidden

    def forward(self, x):
        return self.layers(x)



class Discriminator(torch.nn.Module):
    def __init__(self, latent_dim, in_dim, layers, dist, hidden):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.in_dim = in_dim
        self.layers = layers
        self.dist = dist
        self.hidden = hidden

    def forward(self, x):
        return self.layers(x)

class ReshapeLayer(torch.nn.Module):
    def __init__(self, *args):
        super(ReshapeLayer, self).__init__()
        self.shape = args

    def forward(self, x):
        print(x.shape)
        return x.view(self.shape)
