## project: tf-flownet
### dataset
    flyingchairs
        68616 / 3 = 22872 pairs (image1, image2, flow)
    generate batch
        train: 20000 pairs
        val: 2000 pairs

### flownet_simple
    20170627
        [settings]
            bs: 64
            lr: 1e-3
        20170627_2
            [settings]
                no relu after flow layer, no relu after deconv flow layer
            [result]
