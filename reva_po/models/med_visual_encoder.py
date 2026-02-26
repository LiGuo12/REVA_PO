import torch
import logging
import timm

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        print("visual encoder has position embeddings")
        pos_embed_checkpoint = checkpoint_model['pos_embed'].float()
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def build_vit_b_CXR_mae_CheXpert(vit_ckp_path=None, img_size=224, num_classes=14):
    model = timm.create_model('vit_base_patch16_224', img_size=img_size, global_pool='avg', num_classes=num_classes, drop_rate=0, drop_path_rate=0.2, pretrained=False)
    if vit_ckp_path is not None:
        logging.info(f"Loading image encoder checkpoint from config path: {vit_ckp_path}")
        ckp_dict = torch.load(vit_ckp_path, map_location="cpu", weights_only=False)
        state_dict = ckp_dict['model']
        for key in ['head.weight', 'head.bias']:
            if key in state_dict:
                del state_dict[key]
        # interpolate position embedding
        interpolate_pos_embed(model, state_dict)
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        logging.info(f"Image encoder incompatible keys: {incompatible_keys}")
    else:
        logging.info("No vit_ckp_path provided. Image encoder will NOT load pretrained weights.")
    
    return model
