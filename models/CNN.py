


class UNet3D(nn.Module):
    
    def __init__(self,
                 in_channels,
                 filters,
                 out_channels,
                 kernel_size = 3,
                 block_depth = 2, 
                 pool_size = 2,
                 activation = 'relu',
                 out_activation = 'linear',
                 batch_norm = False,
                 ):
        super().__init__()

        conv_config = dict(kernel_size=kernel_size,
                        depth=block_depth,
                        activation=activation,
                        batch_norm=batch_norm)
        self.levels = len(filters)
        self.encoder_convs = nn.ModuleList([
            ConvBlock3D(in_channels = in_channels if i==0 else filters[i-1],
                        filters = filters[i],
                        **conv_config)
            for i in range(self.levels)
        ])
        self.decoder_convs = nn.ModuleList([
            ConvBlock3D(in_channels = filters[i] + filters[i+1],
                        filters = filters[i],
                        **conv_config)
            for i in reversed(range(self.levels-1))
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv3d(f, f, kernel_size, padding=kernel_size//2)
            for f in filters[:-1]
        ])
        self.out_conv = nn.Conv3d(filters[0], out_channels, kernel_size=1)

        self.maxpools = nn.ModuleList([
            nn.MaxPool3d(kernel_size=pool_size)
            for _ in range(self.levels-1)
        ])
        self.upsamps = nn.ModuleList([
            nn.Upsample(scale_factor=pool_size, mode='trilinear')
            for _ in range(self.levels-1)
        ])
        self.skip_norms = nn.ModuleList([
            nn.BatchNorm3d(f) if batch_norm else nn.Identity()
            for f in filters[:-1]
        ])
        self.act = ACTIVATIONS[activation](inplace=True) if activation.lower() == 'relu' else ACTIVATIONS[activation]()
        self.out_act = ACTIVATIONS[out_activation](inplace=True) if activation.lower() == 'relu' else ACTIVATIONS[activation]()

    def forward(self, x):
        
        x = self.encoder_convs[0](x)
        skips = []
        for conv, pool, skip_conv, skip_norm in zip(self.encoder_convs[1:], self.maxpools, self.skip_convs, self.skip_norms):
            skips.append(skip_norm(self.act(skip_conv(x))))
            x = conv(pool(x))

        skips.reverse()
        for conv, upsamp, skip in zip(self.decoder_convs, self.upsamps, skips):
            x = conv(torch.cat([skip, upsamp(x)], axis=1))
        
        x = self.out_act(self.out_conv(x))

        return x
        