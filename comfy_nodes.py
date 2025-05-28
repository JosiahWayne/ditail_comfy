from comfy.model import SDXL, UNet1, BaseNode
from comfy.prompt_layer import SimpleTextPrompt
from torch import device
from ditail_demo import DitailDemo

class DitailNode(BaseNode):
    TITLE = "Ditail Style Transfer"
    TYPE = "ditail"
    CATEGORY = "Image Processing"
    
    def __init__(self, node_id, graph):
        super().__init__(node_id, graph)
        # 定义输入 socket
        self.add_input('source_model', type='MODEL', default='stablediffusionapi/realistic-vision-v51')
        self.add_input('target_model', type='MODEL', default='stablediffusionapi/realistic-vision-v51')
        self.add_input('image', type='IMAGE')
        self.add_input('pos_prompt', type='PROMPT', default='a photo')
        self.add_input('neg_prompt', type='PROMPT', default='blurry')
        self.add_input('attn_ratio', type='FLOAT', default=0.5)
        self.add_input('conv_ratio', type='FLOAT', default=0.8)
        self.add_input('spl_steps', type='INT', default=50)
        self.add_input('inv_steps', type='INT', default=50)
        self.add_input('mask_type', type='STRING', default='full')
        # 定义输出 socket
        self.add_output('result_image', type='IMAGE')

    def process(self, inputs):
        # 从 inputs 拿参数
        inv_model   = inputs['source_model']
        spl_model   = inputs['target_model']
        pil_img     = inputs['image']          # PIL.Image
        pos_prompt  = inputs['pos_prompt']
        neg_prompt  = inputs['neg_prompt']
        attn_ratio  = inputs['attn_ratio']
        conv_ratio  = inputs['conv_ratio']
        spl_steps   = inputs['spl_steps']
        inv_steps   = inputs['inv_steps']
        mask_type   = inputs['mask_type']

        # 构造 args
        class Args: pass
        args = Args()
        args.inv_model   = inv_model
        args.spl_model   = spl_model
        args.img_path    = None     # we'll feed PIL directly
        args.pos_prompt  = pos_prompt
        args.neg_prompt  = neg_prompt
        args.attn_ratio  = attn_ratio
        args.conv_ratio  = conv_ratio
        args.spl_steps   = spl_steps
        args.inv_steps   = inv_steps
        args.mask        = mask_type
        args.output_dir  = '/tmp'   # 临时输出

        # 初始化 Ditail
        ditail = DitailDemo(args)
        # 直接 encode PIL ➔ latent
        latent = ditail.encode_image(pil_img)
        ditail.load_inv_model()
        # 执行反演
        cond = ditail.extract_latents_from_prompts()
        ditail.invert_image(cond, latent)
        # 执行采样
        ditail.load_spl_model()
        ditail.init_injection(attn_ratio, conv_ratio)
        ditail.sampling_loop()
        # 解码回 PIL
        out_pil = ditail.latent_to_image(ditail.output_latent)

        return {'result_image': out_pil}

# 最后一步：让 ComfyUI 知道这个节点
def register_nodes():
    return [DitailNode]
