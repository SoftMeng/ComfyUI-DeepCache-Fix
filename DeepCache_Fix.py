import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, timestep_embedding, th, apply_control

class DeepCache_Fix:

    @classmethod
    def INPUT_TYPES(s):

        """
        静态方法，用于定义输入参数的类型和默认值。

        该方法返回一个字典，其中包含了不同输入参数的配置信息。每个输入参数都是一个键值对，
        键表示参数名，值是一个元组，包含参数的类型和一个字典，该字典描述了参数的更多细节，
        如默认值、最小值、最大值等。

        返回:
            dict: 包含所有输入参数配置的字典。
        """
        return {
            "required": {
                "model": ("MODEL",),
                "cache_interval": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "cache_depth": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 12,
                    "step": 1,
                    "display": "number"
                }),
                "start_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "end_steps": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "input_cache": (["No", "Yes"], {"default":"Yes"}),
                "middle_cahce": (["No", "Yes"], {"default":"Yes"}),
                "output_cache": (["No", "Yes"], {"default":"Yes"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"

    def apply(self, model, cache_interval, cache_depth, start_steps, end_steps, input_cache, middle_cahce, output_cache):
        # 初始化一些变量
        current_time = -1
        current_step = -1
        model_step = 0
        cache_h = None

        # 创建一个新的模型副本，用于存储修改后的模型。
        new_model = model.clone()
        # 获取并初始化模型的扩散部分。
        unet = new_model.model.diffusion_model
        # 获取并初始化模型的扩散部分。
        dtype = new_model.model.get_dtype()

        def cache_apply_methon(current, start, end):
            """
            判断当前步骤是否在指定的开始步骤和结束步骤范围内。

            参数:
            current -- 当前的步骤数
            start -- 范围的起始步骤数
            end -- 范围的结束步骤数

            返回:
            如果当前步骤在指定范围内，则返回True；否则返回False。
            """
            return start <= current <= end

        def apply_model(model_function, kwargs):
            """
            应用模型函数到给定的输入上。

            这个函数处理模型的输入和输出，包括数据类型转换、条件的添加和模型的分块计算。
            它还处理缓存机制，以在多次调用之间复用计算结果，提高效率。

            :param model_function: 模型函数，一个接受kwargs参数的函数。
            :param kwargs: 包含模型输入和配置的字典。包括输入数据、时间步、条件等。
            :return: 模型处理后的输出。
            """

            # 声明一些非局部变量，用于处理缓存和当前时间步等状态。
            nonlocal model_step, cache_h, current_time, current_step

            # 从kwargs中提取必要的输入和配置。
            xa = kwargs["input"]
            t = kwargs["timestep"]
            c_concat = kwargs["c"].get("c_concat", None)
            c_crossattn = kwargs["c"].get("c_crossattn", None)
            y = kwargs["c"].get("y", None)
            control = kwargs["c"].get("control", None)
            transformer_options = kwargs["c"].get("transformer_options", None)

            # 根据当前时间步计算输入xc。
            sigma = t
            xc = new_model.model.model_sampling.calculate_input(sigma, xa)
            if c_concat is not None:
                # 将输入xc与跨注意力的上下文c_concat进行拼接。
                xc = torch.cat([xc] + [c_concat], dim=1)

            # 处理跨注意力的上下文和数据类型的转换。
            context = c_crossattn
            xc = xc.to(dtype)
            # 将时间步转换为指定的数据类型。
            t = new_model.model.model_sampling.timestep(t).float().to(xc.device)
            context = context.to(dtype=dtype, device=xc.device)

            # 将所有额外的条件转换为指定的数据类型。
            extra_conds = {}
            for o in kwargs:
                extra = kwargs[o]
                if hasattr(extra, "to"):
                    extra = extra.to(dtype)
                extra_conds[o] = extra

            # 初始化模型的输入和配置。
            x = xc
            timesteps = t
            y = None if y is None else y.to(device=x.device, dtype=dtype)
            transformer_options["original_shape"] = list(x.shape)
            transformer_options["current_index"] = 0
            transformer_patches = transformer_options.get("patches", {})

            model_step += 1
            # 更新当前时间步和缓存状态，根据当前时间步决定是否应用模型。
            if t[0].item() > current_time:
                model_step = 0
                current_step = -1
            # 判断是否需要应用模型，根据当前时间步和指定的时间范围。
            cache_apply = cache_apply_methon(model_step, start_steps, end_steps)
            if cache_apply:
                current_step += 1
            else:
                current_step = -1
            current_time = t[0].item()
            # print(f"model_step: {model_step}, {cache_apply}")

            # 确保如果模型是分类的，那么必须提供标签y。
            assert (y is not None) == (
                    unet.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"

            # 处理时间嵌入和模型的输入、中间和输出块。
            hs = []
            t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(unet.dtype)
            emb = unet.time_embed(t_emb)
            if unet.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + unet.label_emb(y)
            xuh = x.type(unet.dtype)
            # current_step 是 cache_interval 的整数倍?
            step_cache_interval = current_step % cache_interval
            # 循环处理输入块。
            for id, module in enumerate(unet.input_blocks):
                transformer_options["block"] = ("input", id)
                xuh = forward_timestep_embed(module, xuh, emb, context, transformer_options)
                xuh = apply_control(xuh, control, 'input')
                if "input_block_patch" in transformer_patches:
                    patch = transformer_patches["input_block_patch"]
                    for p in patch:
                        xuh = p(xuh, transformer_options)
                hs.append(xuh)
                if "input_block_patch_after_skip" in transformer_patches:
                    patch = transformer_patches["input_block_patch_after_skip"]
                    for p in patch:
                        xuh = p(xuh, transformer_options)

                # 根据缓存策略决定是否继续处理或使用缓存。
                if id == cache_depth and cache_apply and input_cache:
                    if not step_cache_interval == 0:
                        break

            # 处理中间块，同样考虑缓存策略。
            # 如果 current_step 是 cache_interval 的整数倍
            # 或者 cache_apply 为 False
            # 或者 middle_cahce 为 False (开关关闭)
            # 则执行中间块的处理。
            if step_cache_interval == 0 or not cache_apply or not middle_cahce:
                transformer_options["block"] = ("middle", 0)
                xuh = forward_timestep_embed(unet.middle_block, xuh, emb, context, transformer_options)
                xuh = apply_control(xuh, control, 'middle')

            # 处理输出块，包括缓存的加载和使用。
            for id, module in enumerate(unet.output_blocks):
                if id < len(unet.output_blocks) - cache_depth - 1 and cache_apply and output_cache:
                    if not step_cache_interval == 0:
                        continue
                if id == len(unet.output_blocks) - cache_depth - 1 and cache_apply and output_cache:
                    if step_cache_interval == 0:
                        cache_h = xuh  # cache
                    else:
                        xuh = cache_h  # load cache
                transformer_options["block"] = ("output", id)
                hsp = hs.pop()
                hsp = apply_control(hsp, control, 'output')
                if "output_block_patch" in transformer_patches:
                    patch = transformer_patches["output_block_patch"]
                    for p in patch:
                        xuh, hsp = p(xuh, hsp, transformer_options)
                xuh = th.cat([xuh, hsp], dim=1)
                del hsp
                if len(hs) > 0:
                    output_shape = hs[-1].shape
                else:
                    output_shape = None
                xuh = forward_timestep_embed(module, xuh, emb, context, transformer_options, output_shape)

            # 将输出转换回原始数据类型，并根据模型配置进行噪声消除计算。
            xuh = xuh.type(x.dtype)
            if unet.predict_codebook_ids:
                model_output = unet.id_predictor(xuh)
            else:
                model_output = unet.out(xuh)

            # 返回计算得到的最终输出。
            return new_model.model.model_sampling.calculate_denoised(sigma, model_output, xa)

        new_model.set_model_unet_function_wrapper(apply_model)

        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "DeepCache_Fix": DeepCache_Fix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepCache_Fix": "DeepCache_Fix",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
