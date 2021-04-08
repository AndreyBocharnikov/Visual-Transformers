def change_names(official_state_dict):
    my_state_dict = {}
    for k, v in official_state_dict.items():
        if k in official_resnet_to_resnetBackbone:
            my_state_dict[official_resnet_to_resnetBackbone[k]] = v
    return my_state_dict


official_resnet_to_resnetBackbone = {
    "conv1.weight": "conv1.0.weight",
    "bn1.weight": "conv1.1.weight",
    "bn1.bias": "conv1.1.bias",
    "bn1.running_mean": "conv1.1.running_mean",
    "bn1.running_var": "conv1.1.running_var",
    "bn1.num_batches_tracked": "conv1.1.num_batches_tracked",
    "layer1.0.conv1.weight": "conv2.0.plain_arch.0.weight",
    "layer1.0.bn1.weight": "conv2.0.plain_arch.1.weight",
    "layer1.0.bn1.bias": "conv2.0.plain_arch.1.bias",
    "layer1.0.bn1.running_mean": "conv2.0.plain_arch.1.running_mean",
    "layer1.0.bn1.running_var": "conv2.0.plain_arch.1.running_var",
    "layer1.0.bn1.num_batches_tracked": "conv2.0.plain_arch.1.num_batches_tracked",
    "layer1.0.conv2.weight": "conv2.0.plain_arch.3.weight",
    "layer1.0.bn2.weight": "conv2.0.plain_arch.4.weight",
    "layer1.0.bn2.bias": "conv2.0.plain_arch.4.bias",
    "layer1.0.bn2.running_mean": "conv2.0.plain_arch.4.running_mean",
    "layer1.0.bn2.running_var": "conv2.0.plain_arch.4.running_var",
    "layer1.0.bn2.num_batches_tracked": "conv2.0.plain_arch.4.num_batches_tracked",
    "layer1.0.conv3.weight": "conv2.0.plain_arch.6.weight",
    "layer1.0.bn3.weight": "conv2.0.plain_arch.7.weight",
    "layer1.0.bn3.bias": "conv2.0.plain_arch.7.bias",
    "layer1.0.bn3.running_mean": "conv2.0.plain_arch.7.running_mean",
    "layer1.0.bn3.running_var": "conv2.0.plain_arch.7.running_var",
    "layer1.0.bn3.num_batches_tracked": "conv2.0.plain_arch.7.num_batches_tracked",
    "layer1.0.downsample.0.weight": "conv2.0.downsample.0.weight",
    "layer1.0.downsample.1.weight": "conv2.0.downsample.1.weight",
    "layer1.0.downsample.1.bias": "conv2.0.downsample.1.bias",
    "layer1.0.downsample.1.running_mean": "conv2.0.downsample.1.running_mean",
    "layer1.0.downsample.1.running_var": "conv2.0.downsample.1.running_var",
    "layer1.0.downsample.1.num_batches_tracked": "conv2.0.downsample.1.num_batches_tracked",
    "layer1.1.conv1.weight": "conv2.1.plain_arch.0.weight",
    "layer1.1.bn1.weight": "conv2.1.plain_arch.1.weight",
    "layer1.1.bn1.bias": "conv2.1.plain_arch.1.bias",
    "layer1.1.bn1.running_mean": "conv2.1.plain_arch.1.running_mean",
    "layer1.1.bn1.running_var": "conv2.1.plain_arch.1.running_var",
    "layer1.1.bn1.num_batches_tracked": "conv2.1.plain_arch.1.num_batches_tracked",
    "layer1.1.conv2.weight": "conv2.1.plain_arch.3.weight",
    "layer1.1.bn2.weight": "conv2.1.plain_arch.4.weight",
    "layer1.1.bn2.bias": "conv2.1.plain_arch.4.bias",
    "layer1.1.bn2.running_mean": "conv2.1.plain_arch.4.running_mean",
    "layer1.1.bn2.running_var": "conv2.1.plain_arch.4.running_var",
    "layer1.1.bn2.num_batches_tracked": "conv2.1.plain_arch.4.num_batches_tracked",
    "layer1.1.conv3.weight": "conv2.1.plain_arch.6.weight",
    "layer1.1.bn3.weight": "conv2.1.plain_arch.7.weight",
    "layer1.1.bn3.bias": "conv2.1.plain_arch.7.bias",
    "layer1.1.bn3.running_mean": "conv2.1.plain_arch.7.running_mean",
    "layer1.1.bn3.running_var": "conv2.1.plain_arch.7.running_var",
    "layer1.1.bn3.num_batches_tracked": "conv2.1.plain_arch.7.num_batches_tracked",
    "layer1.2.conv1.weight": "conv2.2.plain_arch.0.weight",
    "layer1.2.bn1.weight": "conv2.2.plain_arch.1.weight",
    "layer1.2.bn1.bias": "conv2.2.plain_arch.1.bias",
    "layer1.2.bn1.running_mean": "conv2.2.plain_arch.1.running_mean",
    "layer1.2.bn1.running_var": "conv2.2.plain_arch.1.running_var",
    "layer1.2.bn1.num_batches_tracked": "conv2.2.plain_arch.1.num_batches_tracked",
    "layer1.2.conv2.weight": "conv2.2.plain_arch.3.weight",
    "layer1.2.bn2.weight": "conv2.2.plain_arch.4.weight",
    "layer1.2.bn2.bias": "conv2.2.plain_arch.4.bias",
    "layer1.2.bn2.running_mean": "conv2.2.plain_arch.4.running_mean",
    "layer1.2.bn2.running_var": "conv2.2.plain_arch.4.running_var",
    "layer1.2.bn2.num_batches_tracked": "conv2.2.plain_arch.4.num_batches_tracked",
    "layer1.2.conv3.weight": "conv2.2.plain_arch.6.weight",
    "layer1.2.bn3.weight": "conv2.2.plain_arch.7.weight",
    "layer1.2.bn3.bias": "conv2.2.plain_arch.7.bias",
    "layer1.2.bn3.running_mean": "conv2.2.plain_arch.7.running_mean",
    "layer1.2.bn3.running_var": "conv2.2.plain_arch.7.running_var",
    "layer1.2.bn3.num_batches_tracked": "conv2.2.plain_arch.7.num_batches_tracked",
    "layer2.0.conv1.weight": "conv3.0.plain_arch.0.weight",
    "layer2.0.bn1.weight": "conv3.0.plain_arch.1.weight",
    "layer2.0.bn1.bias": "conv3.0.plain_arch.1.bias",
    "layer2.0.bn1.running_mean": "conv3.0.plain_arch.1.running_mean",
    "layer2.0.bn1.running_var": "conv3.0.plain_arch.1.running_var",
    "layer2.0.bn1.num_batches_tracked": "conv3.0.plain_arch.1.num_batches_tracked",
    "layer2.0.conv2.weight": "conv3.0.plain_arch.3.weight",
    "layer2.0.bn2.weight": "conv3.0.plain_arch.4.weight",
    "layer2.0.bn2.bias": "conv3.0.plain_arch.4.bias",
    "layer2.0.bn2.running_mean": "conv3.0.plain_arch.4.running_mean",
    "layer2.0.bn2.running_var": "conv3.0.plain_arch.4.running_var",
    "layer2.0.bn2.num_batches_tracked": "conv3.0.plain_arch.4.num_batches_tracked",
    "layer2.0.conv3.weight": "conv3.0.plain_arch.6.weight",
    "layer2.0.bn3.weight": "conv3.0.plain_arch.7.weight",
    "layer2.0.bn3.bias": "conv3.0.plain_arch.7.bias",
    "layer2.0.bn3.running_mean": "conv3.0.plain_arch.7.running_mean",
    "layer2.0.bn3.running_var": "conv3.0.plain_arch.7.running_var",
    "layer2.0.bn3.num_batches_tracked": "conv3.0.plain_arch.7.num_batches_tracked",
    "layer2.0.downsample.0.weight": "conv3.0.downsample.0.weight",
    "layer2.0.downsample.1.weight": "conv3.0.downsample.1.weight",
    "layer2.0.downsample.1.bias": "conv3.0.downsample.1.bias",
    "layer2.0.downsample.1.running_mean": "conv3.0.downsample.1.running_mean",
    "layer2.0.downsample.1.running_var": "conv3.0.downsample.1.running_var",
    "layer2.0.downsample.1.num_batches_tracked": "conv3.0.downsample.1.num_batches_tracked",
    "layer2.1.conv1.weight": "conv3.1.plain_arch.0.weight",
    "layer2.1.bn1.weight": "conv3.1.plain_arch.1.weight",
    "layer2.1.bn1.bias": "conv3.1.plain_arch.1.bias",
    "layer2.1.bn1.running_mean": "conv3.1.plain_arch.1.running_mean",
    "layer2.1.bn1.running_var": "conv3.1.plain_arch.1.running_var",
    "layer2.1.bn1.num_batches_tracked": "conv3.1.plain_arch.1.num_batches_tracked",
    "layer2.1.conv2.weight": "conv3.1.plain_arch.3.weight",
    "layer2.1.bn2.weight": "conv3.1.plain_arch.4.weight",
    "layer2.1.bn2.bias": "conv3.1.plain_arch.4.bias",
    "layer2.1.bn2.running_mean": "conv3.1.plain_arch.4.running_mean",
    "layer2.1.bn2.running_var": "conv3.1.plain_arch.4.running_var",
    "layer2.1.bn2.num_batches_tracked": "conv3.1.plain_arch.4.num_batches_tracked",
    "layer2.1.conv3.weight": "conv3.1.plain_arch.6.weight",
    "layer2.1.bn3.weight": "conv3.1.plain_arch.7.weight",
    "layer2.1.bn3.bias": "conv3.1.plain_arch.7.bias",
    "layer2.1.bn3.running_mean": "conv3.1.plain_arch.7.running_mean",
    "layer2.1.bn3.running_var": "conv3.1.plain_arch.7.running_var",
    "layer2.1.bn3.num_batches_tracked": "conv3.1.plain_arch.7.num_batches_tracked",
    "layer2.2.conv1.weight": "conv3.2.plain_arch.0.weight",
    "layer2.2.bn1.weight": "conv3.2.plain_arch.1.weight",
    "layer2.2.bn1.bias": "conv3.2.plain_arch.1.bias",
    "layer2.2.bn1.running_mean": "conv3.2.plain_arch.1.running_mean",
    "layer2.2.bn1.running_var": "conv3.2.plain_arch.1.running_var",
    "layer2.2.bn1.num_batches_tracked": "conv3.2.plain_arch.1.num_batches_tracked",
    "layer2.2.conv2.weight": "conv3.2.plain_arch.3.weight",
    "layer2.2.bn2.weight": "conv3.2.plain_arch.4.weight",
    "layer2.2.bn2.bias": "conv3.2.plain_arch.4.bias",
    "layer2.2.bn2.running_mean": "conv3.2.plain_arch.4.running_mean",
    "layer2.2.bn2.running_var": "conv3.2.plain_arch.4.running_var",
    "layer2.2.bn2.num_batches_tracked": "conv3.2.plain_arch.4.num_batches_tracked",
    "layer2.2.conv3.weight": "conv3.2.plain_arch.6.weight",
    "layer2.2.bn3.weight": "conv3.2.plain_arch.7.weight",
    "layer2.2.bn3.bias": "conv3.2.plain_arch.7.bias",
    "layer2.2.bn3.running_mean": "conv3.2.plain_arch.7.running_mean",
    "layer2.2.bn3.running_var": "conv3.2.plain_arch.7.running_var",
    "layer2.2.bn3.num_batches_tracked": "conv3.2.plain_arch.7.num_batches_tracked",
    "layer2.3.conv1.weight": "conv3.3.plain_arch.0.weight",
    "layer2.3.bn1.weight": "conv3.3.plain_arch.1.weight",
    "layer2.3.bn1.bias": "conv3.3.plain_arch.1.bias",
    "layer2.3.bn1.running_mean": "conv3.3.plain_arch.1.running_mean",
    "layer2.3.bn1.running_var": "conv3.3.plain_arch.1.running_var",
    "layer2.3.bn1.num_batches_tracked": "conv3.3.plain_arch.1.num_batches_tracked",
    "layer2.3.conv2.weight": "conv3.3.plain_arch.3.weight",
    "layer2.3.bn2.weight": "conv3.3.plain_arch.4.weight",
    "layer2.3.bn2.bias": "conv3.3.plain_arch.4.bias",
    "layer2.3.bn2.running_mean": "conv3.3.plain_arch.4.running_mean",
    "layer2.3.bn2.running_var": "conv3.3.plain_arch.4.running_var",
    "layer2.3.bn2.num_batches_tracked": "conv3.3.plain_arch.4.num_batches_tracked",
    "layer2.3.conv3.weight": "conv3.3.plain_arch.6.weight",
    "layer2.3.bn3.weight": "conv3.3.plain_arch.7.weight",
    "layer2.3.bn3.bias": "conv3.3.plain_arch.7.bias",
    "layer2.3.bn3.running_mean": "conv3.3.plain_arch.7.running_mean",
    "layer2.3.bn3.running_var": "conv3.3.plain_arch.7.running_var",
    "layer2.3.bn3.num_batches_tracked": "conv3.3.plain_arch.7.num_batches_tracked",
    "layer3.0.conv1.weight": "conv4.0.plain_arch.0.weight",
    "layer3.0.bn1.weight": "conv4.0.plain_arch.1.weight",
    "layer3.0.bn1.bias": "conv4.0.plain_arch.1.bias",
    "layer3.0.bn1.running_mean": "conv4.0.plain_arch.1.running_mean",
    "layer3.0.bn1.running_var": "conv4.0.plain_arch.1.running_var",
    "layer3.0.bn1.num_batches_tracked": "conv4.0.plain_arch.1.num_batches_tracked",
    "layer3.0.conv2.weight": "conv4.0.plain_arch.3.weight",
    "layer3.0.bn2.weight": "conv4.0.plain_arch.4.weight",
    "layer3.0.bn2.bias": "conv4.0.plain_arch.4.bias",
    "layer3.0.bn2.running_mean": "conv4.0.plain_arch.4.running_mean",
    "layer3.0.bn2.running_var": "conv4.0.plain_arch.4.running_var",
    "layer3.0.bn2.num_batches_tracked": "conv4.0.plain_arch.4.num_batches_tracked",
    "layer3.0.conv3.weight": "conv4.0.plain_arch.6.weight",
    "layer3.0.bn3.weight": "conv4.0.plain_arch.7.weight",
    "layer3.0.bn3.bias": "conv4.0.plain_arch.7.bias",
    "layer3.0.bn3.running_mean": "conv4.0.plain_arch.7.running_mean",
    "layer3.0.bn3.running_var": "conv4.0.plain_arch.7.running_var",
    "layer3.0.bn3.num_batches_tracked": "conv4.0.plain_arch.7.num_batches_tracked",
    "layer3.0.downsample.0.weight": "conv4.0.downsample.0.weight",
    "layer3.0.downsample.1.weight": "conv4.0.downsample.1.weight",
    "layer3.0.downsample.1.bias": "conv4.0.downsample.1.bias",
    "layer3.0.downsample.1.running_mean": "conv4.0.downsample.1.running_mean",
    "layer3.0.downsample.1.running_var": "conv4.0.downsample.1.running_var",
    "layer3.0.downsample.1.num_batches_tracked": "conv4.0.downsample.1.num_batches_tracked",
    "layer3.1.conv1.weight": "conv4.1.plain_arch.0.weight",
    "layer3.1.bn1.weight": "conv4.1.plain_arch.1.weight",
    "layer3.1.bn1.bias": "conv4.1.plain_arch.1.bias",
    "layer3.1.bn1.running_mean": "conv4.1.plain_arch.1.running_mean",
    "layer3.1.bn1.running_var": "conv4.1.plain_arch.1.running_var",
    "layer3.1.bn1.num_batches_tracked": "conv4.1.plain_arch.1.num_batches_tracked",
    "layer3.1.conv2.weight": "conv4.1.plain_arch.3.weight",
    "layer3.1.bn2.weight": "conv4.1.plain_arch.4.weight",
    "layer3.1.bn2.bias": "conv4.1.plain_arch.4.bias",
    "layer3.1.bn2.running_mean": "conv4.1.plain_arch.4.running_mean",
    "layer3.1.bn2.running_var": "conv4.1.plain_arch.4.running_var",
    "layer3.1.bn2.num_batches_tracked": "conv4.1.plain_arch.4.num_batches_tracked",
    "layer3.1.conv3.weight": "conv4.1.plain_arch.6.weight",
    "layer3.1.bn3.weight": "conv4.1.plain_arch.7.weight",
    "layer3.1.bn3.bias": "conv4.1.plain_arch.7.bias",
    "layer3.1.bn3.running_mean": "conv4.1.plain_arch.7.running_mean",
    "layer3.1.bn3.running_var": "conv4.1.plain_arch.7.running_var",
    "layer3.1.bn3.num_batches_tracked": "conv4.1.plain_arch.7.num_batches_tracked",
    "layer3.2.conv1.weight": "conv4.2.plain_arch.0.weight",
    "layer3.2.bn1.weight": "conv4.2.plain_arch.1.weight",
    "layer3.2.bn1.bias": "conv4.2.plain_arch.1.bias",
    "layer3.2.bn1.running_mean": "conv4.2.plain_arch.1.running_mean",
    "layer3.2.bn1.running_var": "conv4.2.plain_arch.1.running_var",
    "layer3.2.bn1.num_batches_tracked": "conv4.2.plain_arch.1.num_batches_tracked",
    "layer3.2.conv2.weight": "conv4.2.plain_arch.3.weight",
    "layer3.2.bn2.weight": "conv4.2.plain_arch.4.weight",
    "layer3.2.bn2.bias": "conv4.2.plain_arch.4.bias",
    "layer3.2.bn2.running_mean": "conv4.2.plain_arch.4.running_mean",
    "layer3.2.bn2.running_var": "conv4.2.plain_arch.4.running_var",
    "layer3.2.bn2.num_batches_tracked": "conv4.2.plain_arch.4.num_batches_tracked",
    "layer3.2.conv3.weight": "conv4.2.plain_arch.6.weight",
    "layer3.2.bn3.weight": "conv4.2.plain_arch.7.weight",
    "layer3.2.bn3.bias": "conv4.2.plain_arch.7.bias",
    "layer3.2.bn3.running_mean": "conv4.2.plain_arch.7.running_mean",
    "layer3.2.bn3.running_var": "conv4.2.plain_arch.7.running_var",
    "layer3.2.bn3.num_batches_tracked": "conv4.2.plain_arch.7.num_batches_tracked",
    "layer3.3.conv1.weight": "conv4.3.plain_arch.0.weight",
    "layer3.3.bn1.weight": "conv4.3.plain_arch.1.weight",
    "layer3.3.bn1.bias": "conv4.3.plain_arch.1.bias",
    "layer3.3.bn1.running_mean": "conv4.3.plain_arch.1.running_mean",
    "layer3.3.bn1.running_var": "conv4.3.plain_arch.1.running_var",
    "layer3.3.bn1.num_batches_tracked": "conv4.3.plain_arch.1.num_batches_tracked",
    "layer3.3.conv2.weight": "conv4.3.plain_arch.3.weight",
    "layer3.3.bn2.weight": "conv4.3.plain_arch.4.weight",
    "layer3.3.bn2.bias": "conv4.3.plain_arch.4.bias",
    "layer3.3.bn2.running_mean": "conv4.3.plain_arch.4.running_mean",
    "layer3.3.bn2.running_var": "conv4.3.plain_arch.4.running_var",
    "layer3.3.bn2.num_batches_tracked": "conv4.3.plain_arch.4.num_batches_tracked",
    "layer3.3.conv3.weight": "conv4.3.plain_arch.6.weight",
    "layer3.3.bn3.weight": "conv4.3.plain_arch.7.weight",
    "layer3.3.bn3.bias": "conv4.3.plain_arch.7.bias",
    "layer3.3.bn3.running_mean": "conv4.3.plain_arch.7.running_mean",
    "layer3.3.bn3.running_var": "conv4.3.plain_arch.7.running_var",
    "layer3.3.bn3.num_batches_tracked": "conv4.3.plain_arch.7.num_batches_tracked",
    "layer3.4.conv1.weight": "conv4.4.plain_arch.0.weight",
    "layer3.4.bn1.weight": "conv4.4.plain_arch.1.weight",
    "layer3.4.bn1.bias": "conv4.4.plain_arch.1.bias",
    "layer3.4.bn1.running_mean": "conv4.4.plain_arch.1.running_mean",
    "layer3.4.bn1.running_var": "conv4.4.plain_arch.1.running_var",
    "layer3.4.bn1.num_batches_tracked": "conv4.4.plain_arch.1.num_batches_tracked",
    "layer3.4.conv2.weight": "conv4.4.plain_arch.3.weight",
    "layer3.4.bn2.weight": "conv4.4.plain_arch.4.weight",
    "layer3.4.bn2.bias": "conv4.4.plain_arch.4.bias",
    "layer3.4.bn2.running_mean": "conv4.4.plain_arch.4.running_mean",
    "layer3.4.bn2.running_var": "conv4.4.plain_arch.4.running_var",
    "layer3.4.bn2.num_batches_tracked": "conv4.4.plain_arch.4.num_batches_tracked",
    "layer3.4.conv3.weight": "conv4.4.plain_arch.6.weight",
    "layer3.4.bn3.weight": "conv4.4.plain_arch.7.weight",
    "layer3.4.bn3.bias": "conv4.4.plain_arch.7.bias",
    "layer3.4.bn3.running_mean": "conv4.4.plain_arch.7.running_mean",
    "layer3.4.bn3.running_var": "conv4.4.plain_arch.7.running_var",
    "layer3.4.bn3.num_batches_tracked": "conv4.4.plain_arch.7.num_batches_tracked",
    "layer3.5.conv1.weight": "conv4.5.plain_arch.0.weight",
    "layer3.5.bn1.weight": "conv4.5.plain_arch.1.weight",
    "layer3.5.bn1.bias": "conv4.5.plain_arch.1.bias",
    "layer3.5.bn1.running_mean": "conv4.5.plain_arch.1.running_mean",
    "layer3.5.bn1.running_var": "conv4.5.plain_arch.1.running_var",
    "layer3.5.bn1.num_batches_tracked": "conv4.5.plain_arch.1.num_batches_tracked",
    "layer3.5.conv2.weight": "conv4.5.plain_arch.3.weight",
    "layer3.5.bn2.weight": "conv4.5.plain_arch.4.weight",
    "layer3.5.bn2.bias": "conv4.5.plain_arch.4.bias",
    "layer3.5.bn2.running_mean": "conv4.5.plain_arch.4.running_mean",
    "layer3.5.bn2.running_var": "conv4.5.plain_arch.4.running_var",
    "layer3.5.bn2.num_batches_tracked": "conv4.5.plain_arch.4.num_batches_tracked",
    "layer3.5.conv3.weight": "conv4.5.plain_arch.6.weight",
    "layer3.5.bn3.weight": "conv4.5.plain_arch.7.weight",
    "layer3.5.bn3.bias": "conv4.5.plain_arch.7.bias",
    "layer3.5.bn3.running_mean": "conv4.5.plain_arch.7.running_mean",
    "layer3.5.bn3.running_var": "conv4.5.plain_arch.7.running_var",
    "layer3.5.bn3.num_batches_tracked": "conv4.5.plain_arch.7.num_batches_tracked",
    "layer4.0.conv1.weight": "conv5.0.plain_arch.0.weight",
    "layer4.0.bn1.weight": "conv5.0.plain_arch.1.weight",
    "layer4.0.bn1.bias": "conv5.0.plain_arch.1.bias",
    "layer4.0.bn1.running_mean": "conv5.0.plain_arch.1.running_mean",
    "layer4.0.bn1.running_var": "conv5.0.plain_arch.1.running_var",
    "layer4.0.bn1.num_batches_tracked": "conv5.0.plain_arch.1.num_batches_tracked",
    "layer4.0.conv2.weight": "conv5.0.plain_arch.3.weight",
    "layer4.0.bn2.weight": "conv5.0.plain_arch.4.weight",
    "layer4.0.bn2.bias": "conv5.0.plain_arch.4.bias",
    "layer4.0.bn2.running_mean": "conv5.0.plain_arch.4.running_mean",
    "layer4.0.bn2.running_var": "conv5.0.plain_arch.4.running_var",
    "layer4.0.bn2.num_batches_tracked": "conv5.0.plain_arch.4.num_batches_tracked",
    "layer4.0.conv3.weight": "conv5.0.plain_arch.6.weight",
    "layer4.0.bn3.weight": "conv5.0.plain_arch.7.weight",
    "layer4.0.bn3.bias": "conv5.0.plain_arch.7.bias",
    "layer4.0.bn3.running_mean": "conv5.0.plain_arch.7.running_mean",
    "layer4.0.bn3.running_var": "conv5.0.plain_arch.7.running_var",
    "layer4.0.bn3.num_batches_tracked": "conv5.0.plain_arch.7.num_batches_tracked",
    "layer4.0.downsample.0.weight": "conv5.0.downsample.0.weight",
    "layer4.0.downsample.1.weight": "conv5.0.downsample.1.weight",
    "layer4.0.downsample.1.bias": "conv5.0.downsample.1.bias",
    "layer4.0.downsample.1.running_mean": "conv5.0.downsample.1.running_mean",
    "layer4.0.downsample.1.running_var": "conv5.0.downsample.1.running_var",
    "layer4.0.downsample.1.num_batches_tracked": "conv5.0.downsample.1.num_batches_tracked",
    "layer4.1.conv1.weight": "conv5.1.plain_arch.0.weight",
    "layer4.1.bn1.weight": "conv5.1.plain_arch.1.weight",
    "layer4.1.bn1.bias": "conv5.1.plain_arch.1.bias",
    "layer4.1.bn1.running_mean": "conv5.1.plain_arch.1.running_mean",
    "layer4.1.bn1.running_var": "conv5.1.plain_arch.1.running_var",
    "layer4.1.bn1.num_batches_tracked": "conv5.1.plain_arch.1.num_batches_tracked",
    "layer4.1.conv2.weight": "conv5.1.plain_arch.3.weight",
    "layer4.1.bn2.weight": "conv5.1.plain_arch.4.weight",
    "layer4.1.bn2.bias": "conv5.1.plain_arch.4.bias",
    "layer4.1.bn2.running_mean": "conv5.1.plain_arch.4.running_mean",
    "layer4.1.bn2.running_var": "conv5.1.plain_arch.4.running_var",
    "layer4.1.bn2.num_batches_tracked": "conv5.1.plain_arch.4.num_batches_tracked",
    "layer4.1.conv3.weight": "conv5.1.plain_arch.6.weight",
    "layer4.1.bn3.weight": "conv5.1.plain_arch.7.weight",
    "layer4.1.bn3.bias": "conv5.1.plain_arch.7.bias",
    "layer4.1.bn3.running_mean": "conv5.1.plain_arch.7.running_mean",
    "layer4.1.bn3.running_var": "conv5.1.plain_arch.7.running_var",
    "layer4.1.bn3.num_batches_tracked": "conv5.1.plain_arch.7.num_batches_tracked",
    "layer4.2.conv1.weight": "conv5.2.plain_arch.0.weight",
    "layer4.2.bn1.weight": "conv5.2.plain_arch.1.weight",
    "layer4.2.bn1.bias": "conv5.2.plain_arch.1.bias",
    "layer4.2.bn1.running_mean": "conv5.2.plain_arch.1.running_mean",
    "layer4.2.bn1.running_var": "conv5.2.plain_arch.1.running_var",
    "layer4.2.bn1.num_batches_tracked": "conv5.2.plain_arch.1.num_batches_tracked",
    "layer4.2.conv2.weight": "conv5.2.plain_arch.3.weight",
    "layer4.2.bn2.weight": "conv5.2.plain_arch.4.weight",
    "layer4.2.bn2.bias": "conv5.2.plain_arch.4.bias",
    "layer4.2.bn2.running_mean": "conv5.2.plain_arch.4.running_mean",
    "layer4.2.bn2.running_var": "conv5.2.plain_arch.4.running_var",
    "layer4.2.bn2.num_batches_tracked": "conv5.2.plain_arch.4.num_batches_tracked",
    "layer4.2.conv3.weight": "conv5.2.plain_arch.6.weight",
    "layer4.2.bn3.weight": "conv5.2.plain_arch.7.weight",
    "layer4.2.bn3.bias": "conv5.2.plain_arch.7.bias",
    "layer4.2.bn3.running_mean": "conv5.2.plain_arch.7.running_mean",
    "layer4.2.bn3.running_var": "conv5.2.plain_arch.7.running_var",
    "layer4.2.bn3.num_batches_tracked": "conv5.2.plain_arch.7.num_batches_tracked",
}
#x = torch.rand((2, 3, 224, 224))
#with torch.no_grad():
#    print(np.allclose(resnet50(x).numpy(), my_resnet(x).numpy()))

"""
f = open("official_resnet.txt", "r")
f2 = open("my_resnet.txt", "r")
f3 = open("utils.py", "w")
for i in range(122):
    my_module = f2.readline().strip()
    official_module = f.readline().strip()
    #print()
    f3.write('"' + official_module + '": ' + '"' + my_module + '",\n')
#f.write('\n'.join(map(lambda x: x[1:-1], str(official_resnet.state_dict().keys())[12:-2].split(', '))))

f.close()
f2.close()
f3.close()
"""
"""
official_resnet = models.resnet18(pretrained=True)
my_resnet = ResNet18(1000)
f = open("official_resnet.txt", "w")
f2 = open("my_resnet.txt", "w")
f2.write('\n'.join(map(lambda x: x[1:-1], str(my_resnet.state_dict().keys())[12:-2].split(', '))))
f.write('\n'.join(map(lambda x: x[1:-1], str(official_resnet.state_dict().keys())[12:-2].split(', '))))
f.close()
f2.close()
"""