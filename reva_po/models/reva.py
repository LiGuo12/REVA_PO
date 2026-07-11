import torch
import torch.nn as nn
from reva_po.common.registry import registry
from transformers import AutoProcessor
from reva_po.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from reva_po.models.base_model import BaseModel
from peft import get_peft_model, LoraConfig
from reva_po.models.med_visual_encoder import build_vit_b_CXR_mae_CheXpert
from reva_po.models.outputs import Output

# REVA Stage 3 RL
@registry.register_model("REVA_stage3")
class REVA_stage3(BaseModel): 
    """
    REVA Stage 3 Model
    Stage 3 is the RL stage:
      - base backbone: Qwen2.5-VL
      - trainable part: LoRA and visual merger
      - auxiliary: a frozen CheXpert-style classifier to predict candidate findings
        and inject them into the prompt as weak guidance.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "REVA": "configs/models/REVA.yaml" 
    }
    
    def __init__(
        self,
        max_txt_len=100,
        pretrained_stage2=None,  # stage2 checkpoint path
        pretrained_cls_ckp=None, # classifier checkpoint path
        use_lora=False,
        lora_rank=None,
        lora_alpha=None,
        lora_dropout=None,
        lora_target_modules=None,
        thresholds=None,         # per-class thresholds for predicted categories
        dataset="",              # "mimic" or "iuxray"
        temperature=0.7, 
        top_k=0,
        top_p=0.9,
        min_new_tokens=40,
        repetition_penalty=1.05,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # 1) Load Qwen2.5-VL base model and processor
        # ------------------------------------------------------------
        self.dtype = torch.float16
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.dtype, device_map=None
        )        
        print("Base Model: ", model_name)

        self.qwen_processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        # self.qwen_processor.tokenizer.padding_side = "left" # If inference batch size >1, uncomment this line
        self.qwen_tokenizer = self.qwen_processor.tokenizer
        self.end_sym = self.qwen_processor.tokenizer.eos_token

        # ------------------------------------------------------------
        # 2) Generation (sampling) hyperparameters for RL rollouts
        # ------------------------------------------------------------
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_new_tokens = min_new_tokens
        self.repetition_penalty = repetition_penalty
        self.max_txt_len = max_txt_len
        
        # token ids for the chat template "assistant" prefix (used to locate prompt boundary)
        assistant_token = "<|im_start|>assistant\n"
        self.assistant_token_ids = self.qwen_tokenizer.encode(assistant_token)
        
        # ------------------------------------------------------------
        # 3) Wrap base model with LoRA (PEFT)
        # ------------------------------------------------------------
        self.use_lora = use_lora
        if self.use_lora:
            print("Use LoRA")
            lora_config = LoraConfig(inference_mode=False, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, task_type="CAUSAL_LM", target_modules=lora_target_modules)
            self.peft_model = get_peft_model(self.qwen_model, lora_config)

        # ------------------------------------------------------------
        # 4) Allow gradients for merger modules
        # ------------------------------------------------------------
        for name, param in self.peft_model.named_parameters():
            if "visual.merger" in name:
                param.requires_grad = True

        # ------------------------------------------------------------
        # 5) Category names and threshold configuration for classifier guidance
        # ------------------------------------------------------------
        self.category_names=["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
                             "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
                             "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
        self.thresholds = thresholds
        self.num_categories = len(self.category_names)

        self.dataset = dataset.lower()
        print('load classifer for dataset:', self.dataset)

        # ------------------------------------------------------------
        # 6) Load a frozen medical visual encoder (CheXpert-style) for category prediction
        #    Two variants:
        #      - iuxray: encoder gives features; an extra linear head gives logits
        #      - mimic: encoder directly outputs logits
        # ------------------------------------------------------------
        if self.dataset == "iuxray":
            # encoder without built-in classifier head (num_classes=0)
            self.med_visual_encoder = build_vit_b_CXR_mae_CheXpert(vit_ckp_path=None, img_size=224, num_classes=0)
            
            # classifier head for 14 categories
            feat_dim = getattr(self.med_visual_encoder, 'num_features', 768) 
            self.head = nn.Linear(feat_dim, self.num_categories)

            # load pretrained classifier checkpoint
            if pretrained_cls_ckp is not None:
                cls_ckpt = torch.load(pretrained_cls_ckp, map_location="cpu", weights_only=False) 
                if "model" in cls_ckpt:
                    cls_state_dict = cls_ckpt["model"]
                else:
                    cls_state_dict = cls_ckpt
                new_cls_state_dict = {}
                for k, v in cls_state_dict.items():
                    if k.startswith("med_visual_encoder."):
                        new_k = k[len("med_visual_encoder."):]
                    else:
                        new_k = k
                    new_cls_state_dict[new_k] = v

                cls_msg = self.med_visual_encoder.load_state_dict(new_cls_state_dict, strict=False)
                print("Load cls model: ", cls_msg)

            # classifier head
            head_w = new_cls_state_dict["head.weight"]
            head_b = new_cls_state_dict.get("head.bias", None)
            load_dict = {"weight": head_w}
            if head_b is not None:
                load_dict["bias"] = head_b
            msg_head = self.head.load_state_dict(load_dict, strict=False)
            print("Load cls head:", msg_head)
            del new_cls_state_dict

            # freeze classifier and head
            self.med_visual_encoder.requires_grad_(False)
            self.head.requires_grad_(False)

        elif self.dataset == "mimic":
             # encoder includes classifier head (outputs logits for categories)
            self.med_visual_encoder = build_vit_b_CXR_mae_CheXpert(vit_ckp_path=None, img_size=224)

            # load pretrained classifier checkpoint
            if pretrained_cls_ckp is not None:
                print("Load pretrained cls model:")
                cls_ckpt = torch.load(pretrained_cls_ckp, map_location="cpu", weights_only=False)
                if "model" in cls_ckpt:
                    cls_state_dict = cls_ckpt["model"]
                else:
                    cls_state_dict = cls_ckpt
                new_cls_state_dict = {}
                for k, v in cls_state_dict.items():
                    if k.startswith("med_visual_encoder."):
                        new_k = k[len("med_visual_encoder."):]
                    else:
                        new_k = k
                    new_cls_state_dict[new_k] = v

                cls_msg = self.med_visual_encoder.load_state_dict(new_cls_state_dict, strict=False)
                del new_cls_state_dict
                print("Load cls model: ", cls_msg)

            # freeze classifier    
            self.med_visual_encoder.requires_grad_(False)
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset}")
        
        # ------------------------------------------------------------
        # 7) Load stage2 checkpoint into PEFT model (warm start for RL stage)
        # ------------------------------------------------------------
        if pretrained_stage2:
            try:
                print("Load Pretrained Stage 2 Checkpoint: {}".format(pretrained_stage2))
                pretrained_stage2 = torch.load(pretrained_stage2, map_location="cpu", weights_only=False)
                stage_2_state_dict = pretrained_stage2['model']
                # remap keys so they match PEFT wrapper naming
                new_stage_2_state_dict = {}
                for k, v in stage_2_state_dict.items():
                    if k.startswith("qwen_model.base_model.model."):
                        new_key = k.replace("qwen_model.base_model.model.", "base_model.model.")
                    elif k.startswith("qwen_model_base."):
                        new_key = k.replace("qwen_model_base.", "base_model.model.")
                    else:
                        new_key = k
                    new_stage_2_state_dict[new_key] = v
                msg = self.peft_model.load_state_dict(new_stage_2_state_dict, strict=False)
                print("stage 2 msg: ", msg)
                del new_stage_2_state_dict
            except Exception as e:
                print("Error in loading or setting state_dict:", e)

        # ------------------------------------------------------------
        # 8) Memory-saving training settings
        # ------------------------------------------------------------
        self.qwen_model.gradient_checkpointing_enable()
        self.qwen_model.enable_input_require_grads()

    def build_prompt(self, images, predicted_categories, batch_size):
        """
        Build chat-style messages for Qwen processor.

        Args:
            images:
              - iuxray: list of list-of-images, each sample has 2 views
              - mimic: list of single images
            predicted_categories:
              list[list[str]], length B, predicted label names per sample
            batch_size: B

        Returns:
            messages: list of per-sample chat messages in the format required by Qwen processor
        """
        messages = []
        if self.dataset == "iuxray":
            # IU-Xray has two views (frontal + lateral) per sample
            for i in range(batch_size):
                prompt = (
                    "Generate a diagnosis report for these chest x-ray images (frontal and side views).\n"
                    f"The following findings may be present: {predicted_categories[i]}."
                )
                # attach all views as image blocks + one text block
                content = [{"type": "image", "image": img} for img in images[i]]
                content.append({"type": "text", "text": prompt})
                message = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
                messages.append(message)
        
        elif self.dataset == "mimic":
            # MIMIC-CXR uses a single image per sample
            for i in range(batch_size):
                prompt = (
                    "You are a radiologist. Analyze the input chest X-ray. Assess the major anatomical regions if visible (e.g., airways, lungs, pleura, heart, mediastinum, great vessels, diaphragm, bones, upper abdomen [e.g., liver, stomach], and support devices). Describe any observed findings that are clinically relevant, whether normal or abnormal.\n"
                    f"The following findings may be present: {predicted_categories[i]}."
                )
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": images[i]
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                messages.append(message)
        return messages

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None):
        """ 
        Forward used during training (computing logits for RL loss, KL, etc.)

        Args:
            input_ids: [N, T]
            attention_mask: [N, T]
            pixel_values, image_grid_thw: vision inputs produced by Qwen processor

        Returns:
            outputs: model outputs with logits, etc.
        """        
        outputs = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            return_dict=True,
        )
        return outputs

    def generate(self, samples, mode="default", num_return_sequences=5):
        """
        Generate reports for a batch, optionally with sampling for RL rollouts.

        Args:
            samples: dictionary containing:
                - image: batch of images (B, C, H, W)
                - text_input: list of ground truth texts (B)
            mode: generation mode, "default" or "sample"
            num_return_sequences: only used in "sample" mode; generates G sequences per input
        Returns:
            results: dictionary containing:
                - predicted_reports: list of generated captions (B * num_return_sequences)
                - gt_reports: list of ground truth captions (B)
                - output_ids: tensor of generated token ids (B * num_return_sequences, seq_len)
                - prompt_lens: list of lengths of input prompts (B * num_return_sequences)
                - pixel_values: tensor of input images (B * num_return_sequences, 3, H, W)
                - image_grid_thw: tensor of image grid (B * num_return_sequences, T, H, W)
                - attention_mask: tensor of attention masks (B * num_return_sequences, seq_len)
        """

        images = samples['image']
        gt_texts = samples["text_input"]
        B = len(gt_texts)
        image_cls = samples["image_cls"] 

        # categories = samples.get("categories")
        # ------------------------------------------------------------
        # 1) Predict categories using a frozen classifier to guide prompting
        # ------------------------------------------------------------
        self.med_visual_encoder.eval()
        enc_param = next(self.med_visual_encoder.parameters())

        predicted_categories = []
        if self.dataset == "iuxray":
            self.head.eval()
            views_per_sample = []
            flat_imgs = []
            # flatten all views into a single tensor [B*V, 3, H, W]
            for i in range(B):
                views = [t.to(device=enc_param.device, dtype=enc_param.dtype)
                        for t in image_cls[i]]
                V = len(views)
                views_per_sample.append(V)
                flat_imgs.extend(views)

            assert len(set(views_per_sample)) == 1, f"All samples must have same #views, got {views_per_sample}"
            V = views_per_sample[0]
            x = torch.stack(flat_imgs, dim=0)               # [B*V, 3, H, W]

            with torch.no_grad():
                feats = self.med_visual_encoder(x)          # [B*V, D]
                D = feats.shape[-1]
                feats = feats.view(B, V, D).mean(dim=1)     # [B, D]
                logits = self.head(feats)                   # [B, C]
                pred_probs = torch.sigmoid(logits)          # [B, C]
        
        elif self.dataset == "mimic":
            image_cls = image_cls.to(device=enc_param.device, dtype=enc_param.dtype)
            with torch.no_grad():
                logits = self.med_visual_encoder(image_cls) # [B, C]
                pred_probs = torch.sigmoid(logits)          # [B, C]
        
        # thresholds is broadcast to [B, C]
        thresholds = torch.tensor(self.thresholds, device=pred_probs.device, dtype=pred_probs.dtype).reshape(1, -1)

        pred_labels = (pred_probs > thresholds)      # [B, C], bool

        # map predicted label indices to label names (list of strings per sample)
        for i in range(B):
            idx = pred_labels[i].nonzero(as_tuple=False).squeeze(1).tolist()
            cats = [self.category_names[j] for j in idx]
            predicted_categories.append(cats)

        # ------------------------------------------------------------
        # 2) Build Qwen chat messages with images + category-guided prompt
        # ------------------------------------------------------------
        messages = self.build_prompt(images, predicted_categories, B)
        
        # ------------------------------------------------------------
        # 3) Convert messages into model inputs using Qwen processor
        # ------------------------------------------------------------
        texts = [
            self.qwen_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.qwen_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # move prepared inputs to the model device
        device = next(self.qwen_model.parameters()).device
        inputs = inputs.to(device)

        # ------------------------------------------------------------
        # 4) Choose generation arguments by mode
        # ------------------------------------------------------------
        if mode == "sample":
            # stochastic generation (used for RL rollouts)
            gen_args = dict(
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                min_new_tokens=self.min_new_tokens,
                max_new_tokens=self.max_txt_len,
                num_return_sequences=num_return_sequences,
                repetition_penalty=self.repetition_penalty
            )
        elif mode == "default":
            # default generation (use for inference)
            gen_args = dict(
                min_new_tokens=self.min_new_tokens,
                max_new_tokens=self.max_txt_len,
                repetition_penalty=self.repetition_penalty
            )
        else:
            raise ValueError(f"Unknown generation mode: {mode}")
        
        # ------------------------------------------------------------
        # 5) Run generation
        #   generated_ids: [B * num_return_sequences, seq_len]
        # ------------------------------------------------------------
        generated_ids = self.peft_model.generate(**inputs, **gen_args)
        
        num_return_sequences = gen_args.get("num_return_sequences", 1)

        # ------------------------------------------------------------
        # 6) Compute prompt lengths and expand to match repeated generations
        # ------------------------------------------------------------
        prompt_lens = [len(prompt_len) for prompt_len in inputs.input_ids] # length B

        expanded_prompt_lens = []
        for l in prompt_lens:
            expanded_prompt_lens.extend([l] * num_return_sequences)        # length B*G 

        # attention mask for generated sequence (pad tokens are 0)
        attention_mask = (generated_ids != self.qwen_tokenizer.pad_token_id).long()

        # ------------------------------------------------------------
        # 7) Trim prompt tokens before decoding (decode only the generated part)
        # ------------------------------------------------------------
        generated_ids_trimmed = []
        for i, out_ids in enumerate(generated_ids):
            trimmed = out_ids[expanded_prompt_lens[i]:]
            generated_ids_trimmed.append(trimmed)

        output_text = self.qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return {
            "predicted_reports": output_text,                  # list[str], length B*G
            "gt_reports": gt_texts,                            # list[str], length B
            "output_ids": generated_ids,                       # Tensor [B*G, T]
            "prompt_lens": expanded_prompt_lens,               # list[int], length B*G
            "pixel_values": inputs.get("pixel_values"),        # repeated internally by processor if needed
            "image_grid_thw": inputs.get("image_grid_thw"),
            "attention_mask": attention_mask,                  # Tensor [B*G, T]
            "predicted_categories_list": predicted_categories, # list[list[str]], length B
        }
    
    @classmethod
    def from_config(cls, cfg):
        """
        Build the model from config dict.
        """
        pretrained_stage2 = cfg.get("pretrained_stage2", None)
        pretrained_cls_ckp = cfg.get("pretrained_cls_ckp", None) 

        # LoRA configuration
        use_lora = cfg.get("use_lora", False)
        lora_rank = cfg.get("lora_rank", None)
        lora_alpha = cfg.get("lora_alpha", None)
        lora_dropout = cfg.get("lora_dropout", None)
        lora_target_modules = cfg.get("lora_target_modules", None)

        # classifier thresholds and dataset name
        thresholds = cfg.get("cls_thresholds", None)
        dataset_name = cfg.get("dataset_name", "")
        
        # generation parameters
        temperature = cfg.get("temperature", 0.7)
        top_k = cfg.get("top_k", 0)
        top_p = cfg.get("top_p", 0.9)
        max_txt_len = cfg.get("max_txt_len",100)
        min_new_tokens = cfg.get("min_new_tokens", 40)
        repetition_penalty = cfg.get("repetition_penalty", 1.05)
        
        evaluate = cfg.get("evaluate", False)
        
        model = cls(
            max_txt_len=max_txt_len,
            pretrained_stage2=pretrained_stage2,
            pretrained_cls_ckp=pretrained_cls_ckp,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            thresholds=thresholds,
            dataset=dataset_name,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        print("evaluate:", evaluate)
        # optionally load stage 3 checkpoint for evaluation
        if evaluate:
            ckpt_path = cfg.get("pretrained_stage3", "")  # load pretrained stage 3 model
            if ckpt_path:
                print("Load Stage 3 Checkpoint: {}".format(ckpt_path))
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                msg = model.load_state_dict(ckpt['model'], strict=False)
                if msg.unexpected_keys:
                    print("Stage 3 checkpoint unexpected keys:", msg.unexpected_keys)
                else:
                    print("Stage 3 checkpoint loaded successfully.")
        return model
    
# REVA Stage 2 Classifier-Guided SFT 
@registry.register_model("REVA_stage2")
class REVA_stage2(BaseModel): 
    """
    REVA Stage 2 Model
    Stage 2 is the Classifier-Guided SFT stage:
      - base backbone: Qwen2.5-VL
      - trainable part: LoRA and visual merger
      - auxiliary: a frozen CheXpert-style classifier to predict candidate findings
        and inject them into the prompt as weak guidance.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "REVA": "configs/models/REVA.yaml" 
    }
    
    def __init__(
        self,
        max_txt_len=100,
        pretrained_stage1=None, # stage 1 checkpoint path
        # lora configurations
        use_lora=False,
        lora_rank=None,
        lora_alpha=None,
        lora_dropout=None,
        lora_target_modules=None,
        # cls
        pretrained_cls_ckp=None,
        thresholds=None,         # per-class thresholds for predicted categories
        dataset="",              # "mimic" or "iuxray"
        min_new_tokens=40,
        repetition_penalty=1.05,

    ):
        super().__init__()
        # ------------------------------------------------------------
        # Load Qwen2.5-VL base model and processor
        # ------------------------------------------------------------
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        print("Base Model: ", model_name)
        self.qwen_model_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=None
        )
        
        self.qwen_processor = AutoProcessor.from_pretrained(model_name)
        # self.qwen_processor.tokenizer.padding_side = "left" # If inference batch size >1, uncomment this line
        self.qwen_tokenizer = self.qwen_processor.tokenizer
        self.max_txt_len = max_txt_len
        self.end_sym = self.qwen_processor.tokenizer.eos_token

        # ------------------------------------------------------------
        # Generation hyperparameters for inference
        # ------------------------------------------------------------
        self.min_new_tokens = min_new_tokens
        self.max_txt_len = max_txt_len
        self.repetition_penalty = repetition_penalty

        # token ids for the chat template "assistant" prefix (used to locate prompt boundary)
        assistant_token = "<|im_start|>assistant\n"
        self.assistant_token_ids = self.qwen_tokenizer.encode(assistant_token)
        
        # load pretrain LLM (Merger modules only)
        if pretrained_stage1:
            print("Load Pretrained Stage 1 Checkpoint (Merger Only): {}".format(pretrained_stage1))
            ckpt = torch.load(pretrained_stage1, map_location="cpu", weights_only=False)
            if 'model' in ckpt:
                state_dict = ckpt['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("qwen_model."):
                    new_key = k[len("qwen_model."):]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            msg = self.qwen_model_base.load_state_dict(new_state_dict, strict=False)

        # LoRA
        self.use_lora = use_lora
        if self.use_lora:
            print("Use LoRA")
            lora_config = LoraConfig(inference_mode=False, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, task_type="CAUSAL_LM", target_modules=lora_target_modules)
            self.qwen_model = get_peft_model(self.qwen_model_base, lora_config)
            
        for name, param in self.qwen_model.named_parameters():
            if "visual.merger" in name:
                param.requires_grad = True

        self.category_names=["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                             "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
                             "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

        self.dataset = dataset.lower()
        print('load classifier for dataset:', self.dataset)
        self.thresholds = thresholds
        self.num_categories = len(self.category_names)

        # ------------------------------------------------------------
        # Load a frozen medical visual encoder (CheXpert-style) for category prediction
        #    Two variants:
        #      - iuxray: encoder gives features; an extra linear head gives logits
        #      - mimic: encoder directly outputs logits
        # ------------------------------------------------------------
        if self.dataset == "iuxray":
            # encoder without built-in classifier head (num_classes=0)
            self.med_visual_encoder = build_vit_b_CXR_mae_CheXpert(vit_ckp_path=None, img_size=224, num_classes=0)
            
            # classifier head for 14 categories
            feat_dim = getattr(self.med_visual_encoder, 'num_features', 768) 
            self.head = nn.Linear(feat_dim, self.num_categories)

            # load pretrained classifier checkpoint
            if pretrained_cls_ckp is not None:
                cls_ckpt = torch.load(pretrained_cls_ckp, map_location="cpu", weights_only=False) 
                if "model" in cls_ckpt:
                    cls_state_dict = cls_ckpt["model"]
                else:
                    cls_state_dict = cls_ckpt
                new_cls_state_dict = {}
                for k, v in cls_state_dict.items():
                    if k.startswith("med_visual_encoder."):
                        new_k = k[len("med_visual_encoder."):]
                    else:
                        new_k = k
                    new_cls_state_dict[new_k] = v

                cls_msg = self.med_visual_encoder.load_state_dict(new_cls_state_dict, strict=False)
                print("Load cls model: ", cls_msg)

            # classifier head
            head_w = new_cls_state_dict["head.weight"]
            head_b = new_cls_state_dict.get("head.bias", None)
            load_dict = {"weight": head_w}
            if head_b is not None:
                load_dict["bias"] = head_b
            msg_head = self.head.load_state_dict(load_dict, strict=False)
            print("Load cls head:", msg_head)
            del new_cls_state_dict

            # freeze classifier and head
            self.med_visual_encoder.requires_grad_(False)
            self.head.requires_grad_(False)

        elif self.dataset == "mimic":
             # encoder includes classifier head (outputs logits for categories)
            self.med_visual_encoder = build_vit_b_CXR_mae_CheXpert(vit_ckp_path=None, img_size=224)

            # load pretrained classifier checkpoint
            if pretrained_cls_ckp is not None:
                print("Load pretrained cls model:")
                cls_ckpt = torch.load(pretrained_cls_ckp, map_location="cpu", weights_only=False)
                if "model" in cls_ckpt:
                    cls_state_dict = cls_ckpt["model"]
                else:
                    cls_state_dict = cls_ckpt
                new_cls_state_dict = {}
                for k, v in cls_state_dict.items():
                    if k.startswith("med_visual_encoder."):
                        new_k = k[len("med_visual_encoder."):]
                    else:
                        new_k = k
                    new_cls_state_dict[new_k] = v

                cls_msg = self.med_visual_encoder.load_state_dict(new_cls_state_dict, strict=False)
                del new_cls_state_dict
                print("Load cls model: ", cls_msg)

            # freeze classifier    
            self.med_visual_encoder.requires_grad_(False)
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset}")
        
        # ------------------------------------------------------------
        # Memory-saving training settings
        # ------------------------------------------------------------
        self.qwen_model_base.gradient_checkpointing_enable()
        self.qwen_model_base.enable_input_require_grads()
        
        print("lora use gradient checkpointing: ", self.qwen_model.is_gradient_checkpointing)

    def build_prompt(self, images, predicted_categories, batch_size):
        """
        Build chat-style messages for Qwen processor.

        Args:
            images:
              - iuxray: list of list-of-images, each sample has 2 views
              - mimic: list of single images
            predicted_categories:
              list[list[str]], length B, predicted label names per sample
            batch_size: B

        Returns:
            messages: list of per-sample chat messages in the format required by Qwen processor
        """
        messages = []
        if self.dataset == "iuxray":
            # IU-Xray has two views (frontal + lateral) per sample
            for i in range(batch_size):
                prompt = (
                    "Generate a diagnosis report for these chest x-ray images (frontal and side views).\n"
                    f"The following findings may be present: {predicted_categories[i]}."
                )
                # attach all views as image blocks + one text block
                content = [{"type": "image", "image": img} for img in images[i]]
                content.append({"type": "text", "text": prompt})
                message = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
                messages.append(message)
        
        elif self.dataset == "mimic":
            # MIMIC-CXR uses a single image per sample
            for i in range(batch_size):
                prompt = (
                    "You are a radiologist. Analyze the input chest X-ray. Assess the major anatomical regions if visible (e.g., airways, lungs, pleura, heart, mediastinum, great vessels, diaphragm, bones, upper abdomen [e.g., liver, stomach], and support devices). Describe any observed findings that are clinically relevant, whether normal or abnormal.\n"
                    f"The following findings may be present: {predicted_categories[i]}."
                )
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": images[i]
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                messages.append(message)
        return messages
    
    def forward(self, samples):
        """ 
        Forward pass for training with batch processing support
        Args:
            samples: dictionary containing:
                - image: batch of images (B, C, H, W)
                - text_input: list of ground truth texts (B)
        Returns:
            loss: training loss
        """

        images = samples['image']
        gt_texts = samples["text_input"]
        B = len(gt_texts)
        image_cls = samples["image_cls"]

        # Predict categories using a frozen classifier to guide prompting
        self.med_visual_encoder.eval()
        enc_param = next(self.med_visual_encoder.parameters())

        predicted_categories = []
        if self.dataset == "iuxray":
            self.head.eval()
            views_per_sample = []
            flat_imgs = []
            # flatten all views into a single tensor [B*V, 3, H, W]
            for i in range(B):
                views = [t.to(device=enc_param.device, dtype=enc_param.dtype)
                        for t in image_cls[i]]
                V = len(views)
                views_per_sample.append(V)
                flat_imgs.extend(views)

            assert len(set(views_per_sample)) == 1, f"All samples must have same #views, got {views_per_sample}"
            V = views_per_sample[0]
            x = torch.stack(flat_imgs, dim=0)           # [B*V, 3, H, W]

            with torch.no_grad():
                feats = self.med_visual_encoder(x)          # [B*V, D]
                D = feats.shape[-1]
                feats = feats.view(B, V, D).mean(dim=1)         # [B, D]
                logits = self.head(feats)           # [B, C]
                pred_probs = torch.sigmoid(logits)  # (B, num_categories)
        
        elif self.dataset == "mimic":
            image_cls = image_cls.to(device=enc_param.device, dtype=enc_param.dtype)
            with torch.no_grad():
                logits = self.med_visual_encoder(image_cls) # [B, C]
                pred_probs = torch.sigmoid(logits)          # [B, C]
        
        thresholds = torch.tensor(self.thresholds, device=pred_probs.device, dtype=pred_probs.dtype).reshape(1, -1)
        pred_labels = (pred_probs > thresholds)      # [B, C], bool
        
        # map predicted label indices to label names (list of strings per sample)
        for i in range(B):
            idx = pred_labels[i].nonzero(as_tuple=False).squeeze(1).tolist()
            cats = [self.category_names[j] for j in idx]
            predicted_categories.append(cats)

        # Build Qwen chat messages with images + category-guided prompt
        messages = self.build_prompt(images, predicted_categories, B)

        # Prepare input text for each sample in the batch
        texts = [
            self.qwen_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        # Add gt_text to the text of each sample
        complete_texts = [
            text + gt_text + self.qwen_tokenizer.eos_token 
            for text, gt_text in zip(texts, gt_texts)
        ]
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.qwen_processor(
            text=complete_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.qwen_model.parameters()).device
        inputs = inputs.to(device)

        # Prepare labels - set the non-gt_text labels to -100
        labels = inputs.input_ids.clone()  

        # Process labels for each batch of samples
        for i in range(B):
            # Find the location of the last assistant token
            for idx in range(len(labels[i]) - len(self.assistant_token_ids) + 1):
                if labels[i][idx:idx+len(self.assistant_token_ids)].tolist() == self.assistant_token_ids:
                    last_assistant_pos = idx + len(self.assistant_token_ids)
            # Mark all positions before the assistant token as -100
            labels[i,:last_assistant_pos] = -100
            # Also mark the padding position as -100
            padding_mask = inputs.attention_mask[i] == 0
            labels[i][padding_mask] = -100
            
        outputs = self.qwen_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            use_cache=False,
            return_dict=True,
        )
        loss = outputs.loss

        return {
            "loss": loss
        }

    def generate(self, samples):
        with torch.inference_mode():

            images = samples['image']
            gt_texts = samples["text_input"]
            B = len(gt_texts)
            image_cls = samples["image_cls"] 

            self.med_visual_encoder.eval()

            enc_param = next(self.med_visual_encoder.parameters())
            predicted_categories = []
            if self.dataset == "iuxray":
                self.head.eval()
                views_per_sample = []
                flat_imgs = []
                # flatten all views into a single tensor [B*V, 3, H, W]
                for i in range(B):
                    views = [t.to(device=enc_param.device, dtype=enc_param.dtype)
                            for t in image_cls[i]]
                    V = len(views)
                    views_per_sample.append(V)
                    flat_imgs.extend(views)

                assert len(set(views_per_sample)) == 1, f"All samples must have same #views, got {views_per_sample}"
                V = views_per_sample[0]
                x = torch.stack(flat_imgs, dim=0)           # [B*V, 3, H, W]

                with torch.no_grad():
                    feats = self.med_visual_encoder(x)          # [B*V, D]
                    D = feats.shape[-1]
                    feats = feats.view(B, V, D).mean(dim=1)         # [B, D]
                    logits = self.head(feats)           # [B, C]
                    pred_probs = torch.sigmoid(logits)  # (B, num_categories)
            
            elif self.dataset == "mimic":
                image_cls = image_cls.to(device=enc_param.device, dtype=enc_param.dtype)
                with torch.no_grad():
                    logits = self.med_visual_encoder(image_cls) # [B, C]
                    pred_probs = torch.sigmoid(logits)          # [B, C]
            
            thresholds = torch.tensor(self.thresholds, device=pred_probs.device, dtype=pred_probs.dtype).reshape(1, -1)
            pred_labels = (pred_probs > thresholds)      # [B, C], bool
            
            # map predicted label indices to label names (list of strings per sample)
            for i in range(B):
                idx = pred_labels[i].nonzero(as_tuple=False).squeeze(1).tolist()
                cats = [self.category_names[j] for j in idx]
                predicted_categories.append(cats)

            # Build Qwen chat messages with images + category-guided prompt
            messages = self.build_prompt(images, predicted_categories, B)

            # Preparation for inference
            texts = [
                self.qwen_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.qwen_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            device = next(self.qwen_model.parameters()).device
            inputs = inputs.to(device)
            
            # Inference: Generation of the output
            generated_ids = self.qwen_model.generate(**inputs, min_new_tokens=self.min_new_tokens, max_new_tokens=self.max_txt_len, repetition_penalty=self.repetition_penalty)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        return Output(
            predicted_reports=output_text,
            gt_reports=gt_texts,
            predicted_categories_list=predicted_categories
        )
    
    @classmethod
    def from_config(cls, cfg):
        """
        Build the model from config dict.
        """
        pretrained_stage1 = cfg.get("pretrained_stage1", None)
        pretrained_cls_ckp = cfg.get("pretrained_cls_ckp", None) 
        # LoRA configuration
        use_lora = cfg.get("use_lora", False)
        lora_rank = cfg.get("lora_rank", None)
        lora_alpha = cfg.get("lora_alpha", None)
        lora_dropout = cfg.get("lora_dropout", None)
        lora_target_modules = cfg.get("lora_target_modules", None)

        # classifier thresholds and dataset name
        thresholds = cfg.get("cls_thresholds", None)
        dataset_name = cfg.get("dataset_name", "")

        # generation parameters
        max_txt_len = cfg.get("max_txt_len",100)
        min_new_tokens = cfg.get("min_new_tokens", 40)
        repetition_penalty = cfg.get("repetition_penalty", 1.05)
        
        evaluate = cfg.get("evaluate", False)

        model = cls(
            max_txt_len=max_txt_len,
            pretrained_stage1=pretrained_stage1,
            pretrained_cls_ckp=pretrained_cls_ckp,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            thresholds=thresholds,
            dataset=dataset_name,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        print("evaluate:", evaluate)
        # optionally load stage 2 checkpoint for evaluation
        if evaluate:
            ckpt_path = cfg.get("pretrained_stage2", "")  # load pretrained stage 2 model
            if ckpt_path:
                print("Load Stage 2 Checkpoint: {}".format(ckpt_path))
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                msg = model.load_state_dict(ckpt['model'], strict=False)
                if msg.unexpected_keys:
                    print("Stage 2 checkpoint unexpected keys:", msg.unexpected_keys)
                else:
                    print("Stage 2 checkpoint loaded successfully.")
        return model
    
# REVA Stage 1 Warmup
@registry.register_model("REVA_stage1")
class REVA_stage1(BaseModel): 
    """
    REVA Stage 1 Model
    Stage 1 is the Warmup stage:
      - base backbone: Qwen2.5-VL
      - trainable part: Visual merger MLP modules only
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "REVA": "configs/models/REVA.yaml" 
    }
    
    def __init__(
        self,
        max_txt_len=100,
        pretrained_stage1_mimic=None, # optionally, stage 1 checkpoint path for mimic
        dataset="",              # "mimic" or "iuxray"
        min_new_tokens=40,
        repetition_penalty=1.05,

    ):
        super().__init__()
        # ------------------------------------------------------------
        # Load Qwen2.5-VL base model and processor
        # ------------------------------------------------------------
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        print("Base Model: ", model_name)
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=None
        )
        self.qwen_processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        # self.qwen_processor.tokenizer.padding_side = "left"  # If inference batch size >1, uncomment this line
        self.qwen_tokenizer = self.qwen_processor.tokenizer 
        self.max_txt_len = max_txt_len
        self.end_sym = self.qwen_processor.tokenizer.eos_token
        
        # ------------------------------------------------------------
        # Generation hyperparameters for inference
        # ------------------------------------------------------------
        self.min_new_tokens = min_new_tokens
        self.max_txt_len = max_txt_len
        self.repetition_penalty = repetition_penalty

        # token ids for the chat template "assistant" prefix (used to locate prompt boundary)
        assistant_token = "<|im_start|>assistant\n"
        self.assistant_token_ids = self.qwen_tokenizer.encode(assistant_token)

        self.dataset = dataset.lower()

        # Note: For iuxray dataset, 
        # the stage 1 checkpoint is initialized from the  mimic stage 1 checkpoint.
        # For mimic dataset,
        # the stage 1 checkpoint is initialized from Qwen checkpoint.
        if self.dataset == "iuxray":
            # load pretrain LLM for MIMIC dataset (Merger modules only)
            if pretrained_stage1_mimic:
                print("Load MIMIC Pretrained Stage 1 Checkpoint: {}".format(pretrained_stage1_mimic))
                ckpt = torch.load(pretrained_stage1_mimic, map_location="cpu", weights_only=False)
                if 'model' in ckpt:
                    state_dict = ckpt['model']
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("qwen_model."):
                        new_key = k[len("qwen_model."):]
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                msg = self.qwen_model.load_state_dict(new_state_dict, strict=False)
        
        # Train only the Merger MLP modules
        for name, param in self.qwen_model.named_parameters():
            if "visual.merger.mlp" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Enable gradient checkpointing for memory efficiency
        self.qwen_model.gradient_checkpointing_enable()
        self.qwen_model.enable_input_require_grads()

    def build_prompt(self, images, batch_size):
        """
        Build chat-style messages for Qwen processor.

        Args:
            images:
              - iuxray: list of list-of-images, each sample has 2 views
              - mimic: list of single images
            batch_size: B

        Returns:
            messages: list of per-sample chat messages in the format required by Qwen processor
        """
        messages = []
        if self.dataset == "iuxray":
            # IU-Xray has two views (frontal + lateral) per sample
            for i in range(batch_size):
                prompt = (
                    "Generate a diagnosis report for these chest x-ray images (frontal and side views)."
                )
                # attach all views as image blocks + one text block
                content = [{"type": "image", "image": img} for img in images[i]]
                content.append({"type": "text", "text": prompt})
                message = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
                messages.append(message)
        
        elif self.dataset == "mimic":
            # MIMIC-CXR uses a single image per sample
            for i in range(batch_size):
                prompt = (
                    "You are a radiologist. Analyze the input chest X-ray. Assess the major anatomical regions if visible (e.g., airways, lungs, pleura, heart, mediastinum, great vessels, diaphragm, bones, upper abdomen [e.g., liver, stomach], and support devices). Describe any observed findings that are clinically relevant, whether normal or abnormal."
                )
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": images[i]
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                messages.append(message)
        return messages
    

    def forward(self, samples):
        """ 
        Forward pass for training with batch processing support
        Args:
            samples: dictionary containing:
                - image: batch of images (B, C, H, W)
                - text_input: list of ground truth texts (B)
        Returns:
            loss: training loss
        """

        images = samples["image"]            # List[List[PIL]]
        gt_texts = samples["text_input"]                # List[str]
        B = len(gt_texts)
        assert len(images) == B, "The number of images and text in a batch is inconsistent"

        # Build Qwen chat messages with images
        messages = self.build_prompt(images, B)

        # Prepare input text for each sample in the batch
        texts = [
            self.qwen_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        # Add gt_text to the text of each sample
        complete_texts = [
            text + gt_text + self.qwen_tokenizer.eos_token 
            for text, gt_text in zip(texts, gt_texts)
        ]
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.qwen_processor(
            text=complete_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.qwen_model.parameters()).device
        inputs = inputs.to(device)

        # Prepare labels - set the non-gt_text labels to -100
        labels = inputs.input_ids.clone()

        # Process labels for each batch of samples
        for i in range(B):
            # Find the location of the last assistant token
            for idx in range(len(labels[i]) - len(self.assistant_token_ids) + 1):
                if labels[i][idx:idx+len(self.assistant_token_ids)].tolist() == self.assistant_token_ids:
                    last_assistant_pos = idx + len(self.assistant_token_ids)
            # Mark all positions before the assistant token as -100
            labels[i,:last_assistant_pos] = -100
            # Also mark the padding position as -100
            padding_mask = inputs.attention_mask[i] == 0
            labels[i][padding_mask] = -100
        
        
        outputs = self.qwen_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            use_cache=False,
            return_dict=True,
        )
        loss = outputs.loss

        return {
            "loss": loss
        }

    def generate(self, samples):
        with torch.inference_mode():
            # load image
            images = samples['image']
            gt_texts = samples["text_input"]
            B = len(gt_texts)
            assert len(images) == B, "The number of images and text in a batch is inconsistent"

            # Build Qwen chat messages with images
            messages = self.build_prompt(images, B)
            
            # Preparation for inference
            texts = [
                self.qwen_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.qwen_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            device = next(self.qwen_model.parameters()).device
            inputs = inputs.to(device)
            # Inference: Generation of the output
            generated_ids = self.qwen_model.generate(**inputs, min_new_tokens=self.min_new_tokens, max_new_tokens=self.max_txt_len, repetition_penalty=self.repetition_penalty)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        return Output(
            predicted_reports=output_text,
            gt_reports=gt_texts
        )
    
    @classmethod
    def from_config(cls, cfg):
        """
        Build the model from config dict.
        """
        pretrained_stage1_mimic = cfg.get("pretrained_stage1_mimic", None)

        dataset_name = cfg.get("dataset_name", "")
        max_txt_len = cfg.get("max_txt_len",100)
        min_new_tokens = cfg.get("min_new_tokens", 40)
        repetition_penalty = cfg.get("repetition_penalty", 1.05)
        evaluate = cfg.get("evaluate", False)

        model = cls(
            max_txt_len=max_txt_len,
            pretrained_stage1_mimic=pretrained_stage1_mimic,
            dataset=dataset_name,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        print("evaluate:", evaluate)
        # optionally load stage 1 checkpoint for evaluation
        if evaluate:
            ckpt_path = cfg.get("pretrained_stage1", "")  # load pretrained stage 1 model
            if ckpt_path:
                print("Load Stage 1 Checkpoint: {}".format(ckpt_path))
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                msg = model.load_state_dict(ckpt['model'], strict=False)
                if msg.unexpected_keys:
                    print("Stage 1 checkpoint unexpected keys:", msg.unexpected_keys)
                else:
                    print("Stage 1 checkpoint loaded successfully.")
        return model
    
# train classifier
@registry.register_model("Medical_MAE")
class Medical_MAE(BaseModel): 
    """
    Medical MAE model.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "Medical_MAE": "configs/models/Medical_MAE.yaml" 
    }
    
    def __init__(
        self,
        vit_ckpt=None,
        pretrained_cls_ckpt_mimic=None,
        dataset="",              # "mimic" or "iuxray"

    ):
        super().__init__()
        # ------------------------------------------------------------
        # Load Medical Visual Encoder
        # ------------------------------------------------------------
        self.category_names=["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
                             "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
                             "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
            
        self.num_categories = len(self.category_names)
        self.category_to_idx = {name: idx for idx, name in enumerate(self.category_names)}
        
        self.dataset = dataset.lower()
        print('dataset:', self.dataset)
        if vit_ckpt is not None:
            print('load medical visual encoder')
            if self.dataset == "iuxray":
                self.med_visual_encoder = build_vit_b_CXR_mae_CheXpert(vit_ckp_path=vit_ckpt, img_size=224, num_classes=0)
                # cls head
                feat_dim = getattr(self.med_visual_encoder, 'num_features', 768) 
                self.head = nn.Linear(feat_dim, self.num_categories)
            elif self.dataset == "mimic":
                self.med_visual_encoder = build_vit_b_CXR_mae_CheXpert(vit_ckp_path=vit_ckpt, img_size=224)
            else:
                raise ValueError(f"Unknown dataset name: {self.dataset}")
        # Note: For iuxray dataset, 
        # the classifier checkpoint is initialized from the mimic classifier checkpoint.
        # For mimic dataset,
        # the classifier checkpoint is initialized from Medical_MAE checkpoint.
        if self.dataset == "iuxray":
            if pretrained_cls_ckpt_mimic is not None:
                print("Load MIMIC pretrained cls model")
                ckpt = torch.load(pretrained_cls_ckpt_mimic, map_location="cpu", weights_only=False) 
                if "model" in ckpt:
                    state_dict = ckpt["model"]
                else:
                    state_dict = ckpt
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("med_visual_encoder."):
                        new_k = k[len("med_visual_encoder."):]
                    else:
                        new_k = k
                    new_state_dict[new_k] = v
                msg = self.med_visual_encoder.load_state_dict(new_state_dict, strict=False)
                print("Load MIMIC pretrained cls msg: ", msg)

            # ---- cls head ----
            head_w = new_state_dict["head.weight"]
            head_b = new_state_dict.get("head.bias", None)
            load_dict = {"weight": head_w}
            if head_b is not None:
                load_dict["bias"] = head_b
            msg_head = self.head.load_state_dict(load_dict, strict=False)
            print("Load MIMIC cls head:", msg_head)

        self.criterion = torch.nn.BCEWithLogitsLoss()

    
    def _categories_to_onehot(self, categories_list):
        """Convert list of category names to one-hot tensors.
        
        Args:
            categories_list: List of lists, where each inner list contains category names
                            for a sample. E.g., [["Lung Opacity", "Pneumonia"], ["No Finding"], ...]
        
        Returns:
            one_hot: Tensor of shape (batch_size, num_categories)
        """
        batch_size = len(categories_list)
        one_hot = torch.zeros(batch_size, self.num_categories)
        
        for i, cats in enumerate(categories_list):
            if cats:  # If list is not empty
                for cat in cats:
                    if cat in self.category_to_idx:
                        idx = self.category_to_idx[cat]
                        one_hot[i, idx] = 1.0
                    else:
                        print(f"Warning: Unknown category '{cat}'")
        
        return one_hot

    def forward(self, samples):
        image_cls = samples["image_cls"]            # length B, each is list of V tensors
        categories = samples.get("categories")      # length B, each is list[str]
        B = len(image_cls)
        device = next(self.med_visual_encoder.parameters()).device
        dtype = next(self.med_visual_encoder.parameters()).dtype
        batch_cats_onehot = self._categories_to_onehot(categories).to(device)
        
        if self.dataset == "iuxray":
            flat_imgs = []
            views_per_sample = []
            for i in range(B):
                views = [t.to(device=device, dtype=dtype)
                        for t in image_cls[i]]
                V = len(views)
                views_per_sample.append(V)
                flat_imgs.extend(views)

            assert len(set(views_per_sample)) == 1, f"All samples must have same #views, got {views_per_sample}"
            V = views_per_sample[0]

            x = torch.stack(flat_imgs, dim=0)           # [B*V, 3, H, W]

            # ViT features: Since num_classes=0 and global_pool='avg', the output is features [B*V, D]
            feats = self.med_visual_encoder(x)          # [B*V, D]
            D = feats.shape[-1]
            feats = feats.view(B, V, D)                 # [B, V, D]

            # Multi-view average pooling
            feats_agg = feats.mean(dim=1)               # [B, D]

            # cls head
            logits = self.head(feats_agg)           # [B, C]
        elif self.dataset == "mimic":
            logits = self.med_visual_encoder(image_cls.to(device=device, dtype=dtype))
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset}")
        
        loss = self.criterion(logits, batch_cats_onehot)
        return {"loss": loss}
    
    @torch.no_grad()
    def generate(self, samples, threshold=0.3):
        """
        Inference for medical image classification.
        Args:
            samples: dict, includes "image_cls"
            threshold: float, the threshold for positive class determination (default 0.3)

        Returns:
            pred_probs: (batch_size, num_categories) probabilities
            pred_labels: (batch_size, num_categories) 0/1, determined by threshold
        """
        self.eval() 
        image_cls = samples["image_cls"]            # length B, each is list of V tensors
        categories = samples.get("categories")      # length B, each is list[str]
        B = len(image_cls)
        device = next(self.med_visual_encoder.parameters()).device
        dtype = next(self.med_visual_encoder.parameters()).dtype
        if self.dataset == "iuxray":
            flat_imgs = []
            views_per_sample = []
            for i in range(B):
                views = [t.to(device=device, dtype=dtype)
                        for t in image_cls[i]]
                V = len(views)
                views_per_sample.append(V)
                flat_imgs.extend(views)

            assert len(set(views_per_sample)) == 1, f"All samples must have same #views, got {views_per_sample}"
            V = views_per_sample[0]

            x = torch.stack(flat_imgs, dim=0)           # [B*V, 3, H, W]

            # Since num_classes=0 and global_pool='avg', the output is features [B*V, D]
            feats = self.med_visual_encoder(x)          # [B*V, D]
            D = feats.shape[-1]
            feats = feats.view(B, V, D)                 # [B, V, D]

            # Multi-view average pooling
            feats_agg = feats.mean(dim=1)               # [B, D]

            # cls head
            logits = self.head(feats_agg)           # [B, C]
        
        elif self.dataset == "mimic":
            logits = self.med_visual_encoder(image_cls.to(device=device, dtype=dtype))

        # probability of each category
        pred_probs = torch.sigmoid(logits)  # (B, num_categories)
        
        # predicted labels based on threshold
        pred_labels = (pred_probs > threshold).long()

        # Convert the ground truth to a one-hot tensor (if needed)
        if isinstance(categories, list):
            gt_labels = self._categories_to_onehot(categories).to(device)
        else:
            gt_labels = categories.to(device)

        return {
            "probs": pred_probs.cpu(),
            "pred_labels": pred_labels.cpu(),
            "gt_labels": gt_labels.cpu()
        }

    @classmethod
    def from_config(cls, cfg):
        model = cls(
            vit_ckpt=cfg.get("vit_ckpt", None),     
            pretrained_cls_ckpt_mimic=cfg.get("pretrained_cls_ckpt_mimic", None),    
            dataset=cfg.get("dataset_name", "")
        )
        return model

