export WANDB_API_KEY=60a6de95b1e6fc7f42362b9acec5c5578f4fde9f

python main.py --name test --model AANet_weight \
--train_root "/home/syx/artifact/dataset/colon_private/train" \
--val_root "/home/syx/artifact/dataset/colon_private/val" \
--test_root "/home/syx/artifact/dataset/colon_private/test" \
--batch_size 8 \
--num_workers 16 \
--rgb_yaml_add "/home/syx/artifact/dataset/colon_private/rsna_yaml/colon_private_RGB_1.yaml" \
--hsv_yaml_add "/home/syx/artifact/dataset/colon_private/rsna_yaml/colon_private_HSV_1.yaml" \
--hed_yaml_add "/home/syx/artifact/dataset/colon_private/rsna_yaml/colon_private_HED_1.yaml" \
--lab_yaml_add "/home/syx/artifact/dataset/colon_private/rsna_yaml/colon_private_LAB_1.yaml" \
--rgb_yaml_add_0 "/home/syx/artifact/dataset/colon_private/rsna_yaml/colon_private_RGB_0.yaml" \
--hsv_yaml_add_0 "/home/syx/artifact/dataset/colon_private/rsna_yaml/colon_private_HSV_0.yaml" \
--hed_yaml_add_0 "/home/syx/artifact/dataset/colon_private/rsna_yaml/colon_private_HED_0.yaml" \
--lab_yaml_add_0 "/home/syx/artifact/dataset/colon_private/rsna_yaml/colon_private_LAB_0.yaml" \
--ncolorspace 1 --rgb 1 --hsv 0 --hed 0 --lab 0 \
--nlayer 2 \
--switch "add"
